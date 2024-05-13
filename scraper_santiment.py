import pandas as pd
import san
from san import AsyncBatch
from pathlib import Path
import os
from datetime import datetime, timezone, timedelta
import keys
import argparse
import json
from apscheduler.schedulers.blocking import BlockingScheduler
import warnings

class SantimentScraper:
    INTERVALS = {'1h': 60, '24h': 1440}
    COINS_DICT = {"BAT": "basic-attention-token", "FTM": "fantom", "ETH": "ethereum",  "BTC": "bitcoin", "USDT": "tether"}
    LOOKBACK = 168 # hours
    METRICS_24H_DELAY = 4 # hours

    def __init__(self, coins, resolutions, start_time, end_time, endpoint_file_paths, save_folder, mode):
        self.coins = coins
        self.resolutions=resolutions
        self.start_time=start_time
        self.end_time=end_time
        self.save_folder = Path(save_folder)
        self.endpoint_file_paths = endpoint_file_paths
        self.endpoints_df = self.load_endpoints()
        self.mode=mode

    def load_endpoints(self):
        endpoints = {}
        for coin in self.coins:
            endpoints[coin] = {}
            for resolution in self.resolutions:
                df = pd.read_csv(self.endpoint_file_paths[coin][resolution], index_col=0)
                endpoints[coin][resolution] = df
        return endpoints

    def save_data_to_csv(self, df, coin, resolution, endpoint):
        folder_path = self.save_folder / coin / resolution
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f"{endpoint}_{coin}_{resolution}_{self.file_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"
        df.to_csv(folder_path / filename)

    def round_to_latest_four_am(self, dt):
        # 1 day metrics from Santiment are scrapped at 4 am next day to reassure metric value is not changing
        return (dt - timedelta(hours=dt.hour - self.METRICS_24H_DELAY if dt.hour >= self.METRICS_24H_DELAY else dt.hour + (24 - self.METRICS_24H_DELAY))).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=1)

    def scrape(self):
        if self.mode=="live":
            current_time = datetime.utcnow()
            file_time = current_time.replace(second=0, microsecond=0)

            self.end_time = current_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            self.start_time = self.end_time - timedelta(hours=self.LOOKBACK+48) # 1 week plus two more days

            self.start_time_human = self.start_time.strftime('%Y-%m-%d %H:%M:%S')
            self.end_time_human = self.end_time.strftime('%Y-%m-%d %H:%M:%S')
            
        if self.mode=="historical":
            file_time = self.end_time 
            self.start_time_human = self.start_time.strftime('%Y-%m-%d %H:%M:%S')
            self.end_time_human = self.end_time.strftime('%Y-%m-%d %H:%M:%S')

        self.file_time = file_time.replace(minute=(file_time.minute // 5) * 5, second=0, microsecond=0)

        for coin in self.coins:
            for resolution in self.resolutions:
                metrics = sorted(self.endpoints_df[coin][resolution]["0"].tolist())

                batch = AsyncBatch()

                if resolution == "1h":
                    for metric in metrics:
                        batch.get(metric, slug=self.COINS_DICT[coin], from_date=self.start_time_human, to_date=self.end_time_human, interval=resolution)

                # Santiment 1 day metrics should be collected at 4 am next day, to ensure all stored metrics are immutable
                if resolution == "24h":    
                    for metric in metrics:
                        batch.get(metric, slug=self.COINS_DICT[coin], from_date=self.start_time_human, to_date=self.round_to_latest_four_am(self.end_time).strftime('%Y-%m-%d %H:%M:%S'), interval=resolution)   

                batch_pd = batch.execute(max_workers=8)

                placeholder = pd.DataFrame(index=pd.date_range(start=self.start_time_human, end=self.end_time_human, freq="1h", inclusive="left"))
                placeholder.index = placeholder.index.tz_localize(None)

                for i, item in enumerate(batch_pd):
                    item.index = item.index.tz_localize(None)
                    item.columns = [metrics[i]]

                    if self.mode=="live":
                        if resolution == "24h":
                            item.index += timedelta(hours=24 + self.METRICS_24H_DELAY)

                        placeholder = placeholder.merge(item.asfreq('H', method='ffill'), how='left', left_index=True, right_index=True)
                    
                    if self.mode=="historical":
                        placeholder = placeholder.combine_first(item)

                    self.save_data_to_csv(item, coin, resolution, metrics[i])

                placeholder = placeholder.fillna(method='ffill') #.fillna(0)
                
                if self.mode=="live":
                    placeholder = placeholder.iloc[-self.LOOKBACK:] # 1 last week of data

                placeholder.index = placeholder.index.tz_localize(None)
                placeholder.columns = [column+f'_{self.INTERVALS[resolution]:04d}_{coin}' for column in metrics]

                placeholder.to_csv(os.path.join(self.save_folder, f"scraped_santiment_{coin}_{resolution}_{self.file_time.strftime('%Y-%m-%d_%H:%M:%S')}.csv"))

    def run_periodic_scrape(self):
        scheduler = BlockingScheduler()
        scheduler.add_job(lambda: self.scrape(), 'cron', hour='*/1') # minute='*/1'
        scheduler.start()


def main(args):
    LOGIN_CREDENTIALS = {"santiment_api_key": keys.san_api_key}
    
    san.ApiConfig.api_key = LOGIN_CREDENTIALS["santiment_api_key"]

    # Set up the folder for saving data
    save_folder_path = Path(args.save_folder)
    save_folder_path.mkdir(parents=True, exist_ok=True)

    f = open(args.endpoint_file_paths)
    endpoint_file_paths = json.load(f)

    if args.mode=="live":
        start_time = end_time = None
    
    if args.mode=="historical":
        start_time = datetime.fromisoformat(args.start_time)
        end_time = datetime.fromisoformat(args.end_time)

    scraper = SantimentScraper(
        coins=args.coins.split(','),
        resolutions=args.resolutions.split(','),
        start_time=start_time,
        end_time=end_time,
        endpoint_file_paths=endpoint_file_paths,
        save_folder=save_folder_path,
        mode=args.mode,
    )

    if args.mode=="live":
        scraper.run_periodic_scrape()

    if args.mode=="historical":
        scraper.scrape()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Santiment data scraper with specified parameters.')
    parser.add_argument('--coins', required=True, help='Comma-separated list of coin names, e.g.,"BTC,ETH,USDT,FTM,BAT"')
    parser.add_argument('--resolutions', required=True, help='Comma-separated list of resolutions, e.g.,"1h,24h"')
    parser.add_argument('--start_time', required=False, help='Start datetime in ISO format, e.g., "2020-07-01T00:00:00", not used for live data')
    parser.add_argument('--end_time', required=False, help='End datetime in ISO format, e.g., "2024-04-11T00:00:00", not used for live data')
    parser.add_argument('--endpoint_file_paths', required=True, help='path of endpoints_file_path_santiment.json')
    parser.add_argument('--save_folder', required=True, help='Folder path to save the scraped data, e.g., "./data/test/santiment/historical"')
    parser.add_argument('--mode', required=True, help='mode to scrape data', choices=["historical", "live"])

    args = parser.parse_args()
    main(args)