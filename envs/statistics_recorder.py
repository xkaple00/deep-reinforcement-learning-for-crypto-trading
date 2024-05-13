from typing import Dict, Any

class StatisticsRecorder:
    def __init__(self, record_statistics: bool = True) -> None:
        self.record_statistics = record_statistics
        
        self.buy_markers_time_short = []
        self.sell_markers_time_short = []
        self.markers_amount_short = []

        self.buy_markers_time_long = []
        self.sell_markers_time_long = []
        self.markers_amount_long = []

        self.reward_realized_pnl_short = []
        self.reward_realized_pnl_long = []

        self.unrealized_pnl_short = []
        self.unrealized_pnl_long = []

        self.equity_list = []
        self.wallet_balance_list = []
        self.action_list = []
        self.reward_list = []

        self.average_price_short_list = []
        self.average_price_long_list = []

    def update(
        self,
        action,
        reward,
        reward_realized_pnl_short,
        reward_realized_pnl_long,
        unrealized_pnl_short,
        unrealized_pnl_long,
        margin_short_start,
        margin_short_end,
        margin_long_start,
        margin_long_end,
        num_steps,
        coins_short,
        coins_long,
        equity,
        wallet_balance,
        average_price_short,
        average_price_long        
    ) -> None:
        if self.record_statistics:
            margin_short_delta = margin_short_end - margin_short_start
            margin_long_delta = margin_long_end - margin_long_start
    
            if margin_short_delta > 0:
                self.sell_markers_time_short.append(num_steps)
                self.markers_amount_short.append(coins_short)
    
            if margin_short_delta < 0:
                self.buy_markers_time_short.append(num_steps)
                self.markers_amount_short.append(coins_short)
    
            if margin_long_delta < 0:
                self.sell_markers_time_long.append(num_steps)
                self.markers_amount_long.append(coins_long)
    
            if margin_long_delta > 0:
                self.buy_markers_time_long.append(num_steps)
                self.markers_amount_long.append(coins_long)
    
            self.reward_realized_pnl_short.append(reward_realized_pnl_short)
            self.reward_realized_pnl_long.append(reward_realized_pnl_long)

            self.unrealized_pnl_short.append(unrealized_pnl_short)
            self.unrealized_pnl_long.append(unrealized_pnl_long)
    
            self.equity_list.append(equity)
            self.wallet_balance_list.append(wallet_balance)
            self.action_list.append(action)
            self.reward_list.append(reward)

            self.average_price_short_list.append(average_price_short)
            self.average_price_long_list.append(average_price_long)


    def get(self) -> Dict[str, Any]:
        info = {}

        if self.record_statistics:
            info = {
                "buy_markers_time_short": self.buy_markers_time_short,
                "sell_markers_time_short": self.sell_markers_time_short,
                "buy_markers_time_long": self.buy_markers_time_long,
                "sell_markers_time_long": self.sell_markers_time_long,
                "markers_amount_short": self.markers_amount_short,
                "markers_amount_long": self.markers_amount_long,
                "reward_realized_pnl_short": self.reward_realized_pnl_short,
                "reward_realized_pnl_long": self.reward_realized_pnl_long,
                "unrealized_pnl_short": self.unrealized_pnl_short,
                "unrealized_pnl_long": self.unrealized_pnl_long,
                "equity": self.equity_list,
                "wallet_balance": self.wallet_balance_list,
                "action": self.action_list,
                "reward": self.reward_list,
                "average_price_short": self.average_price_short_list,
                "average_price_long": self.average_price_long_list,
            }
    
        return info
