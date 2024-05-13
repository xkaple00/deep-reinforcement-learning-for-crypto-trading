import tensorflow as tf
from tensorflow import keras
import os
from keras import Model
from keras.layers import Input, Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add
from tensor_annotations.tensorflow import Tensor0, Tensor1, Tensor2, Tensor3
from tensor_annotations.tensorflow import float32, int32
from tensor_annotations.axes import Axis, Batch, Time, Depth
from typing import NewType, Tuple, Dict, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.typing import TensorType


Const1 = NewType('Const1', Axis)

DFlatHistory = NewType('DFlatHistory', Axis)

DObsTime = NewType('DTime', Axis)
DObsAccount = NewType('DAccount', Axis)
DObsCandlesticksBTC = NewType('DCandlesticksBTC', Axis)
DObsCandlesticksFTM = NewType('DCandlesticksFTM', Axis)
DObsSantimentBTC1h = NewType('DObsSantimentBTC1h', Axis)
DObsSantimentBTC1d = NewType('DObsSantimentBTC1d', Axis)
DObsSantimentFTM1h = NewType('DObsSantimentFTM1h', Axis)
DObsSantimentFTM1d = NewType('DObsSantimentFTM1d', Axis)

DObs = NewType('DObs', Axis)
DObsInternalEnc = NewType('DObsEncInternal', Axis)
DObsExternalEnc = NewType('DObsEncExternal', Axis)
DObsStemEnc = NewType('DObsEnc', Axis)
DFeedForwardInter = NewType('DFeedForwardInter', Axis)

DTimeConcat = NewType('DTimeEncodingBase', Axis)
DTimeEnc = NewType('DTimeEnc', Axis)

DAccountEnc = NewType('DAccountEnc', Axis)

DCandlesticksBTC = NewType('DCandlesticksBTC', Axis)
DCandlesticksFTM = NewType('DCandlesticksFTM', Axis)
DCandlesticksEncBTC = NewType('DCandlesticksEncBTC', Axis)
DCandlesticksEncFTM = NewType('DCandlesticksEncFTM', Axis)
DCandlesticksEncConcat = NewType('DCandlesticksEncConcat', Axis)
DCandlesticksEnc = NewType('DCandlesticksEnc', Axis)

DSantimentBTC = NewType('DSantimentBaseBTC', Axis)
DSantimentFTM = NewType('DSantimentBaseFTM', Axis)
DSantimentEncBTC = NewType('DSantimentEncBTC', Axis)
DSantimentEncFTM = NewType('DSantimentEncFTM', Axis)
DSantimentEncConcat = NewType('DSantimentEncConcat', Axis)
DSantimentEnc = NewType('DSantimentEnc', Axis)

DObsTranfEnc = NewType('DOutputTranfEnc', Axis)

NOutputs = NewType('NOutputs', Axis)


class InputSplit(Layer):
    def __init__(self, num_obs_in_history: int, d_obs: int, d_obs_logical_segments: List[int], name='input_split'):
        super().__init__(name=name)

        self.num_obs_in_history = num_obs_in_history
        self.d_obs = d_obs
        self.d_obs_logical_segments = d_obs_logical_segments

    def call(
            self,
            obs_history_flat: Tensor3[float32, Batch, Time, DFlatHistory]
        ) -> Tuple[
            Tensor3[float32, Batch, Time, DObsTime],
            Tensor3[float32, Batch, Time, DObsAccount],
            Tensor3[float32, Batch, Time, DObsCandlesticksBTC],
            Tensor3[float32, Batch, Time, DObsCandlesticksFTM],
            Tensor3[float32, Batch, Time, DObsSantimentBTC1h],
            Tensor3[float32, Batch, Time, DObsSantimentBTC1d],
            Tensor3[float32, Batch, Time, DObsSantimentFTM1h],
            Tensor3[float32, Batch, Time, DObsSantimentFTM1d]
        ]:

        b_size: Tensor0[int32] = tf.shape(obs_history_flat)[0]

        obs_history: Tensor3[float32, Batch, Time, DObs] = tf.reshape(
            obs_history_flat, shape=[b_size, self.num_obs_in_history, self.d_obs]
        )

        obs_logical_segments: List[Tensor3[float32, Batch, Time, Depth]] = tf.split(
            obs_history, num_or_size_splits=self.d_obs_logical_segments, axis=-1
        )

        obs_history_time: Tensor3[float32, Batch, Time, DObsTime] = obs_logical_segments[0]
        obs_history_account: Tensor3[float32, Batch, Time, DObsAccount] = obs_logical_segments[1]
        obs_history_candlesticks_btc: Tensor3[float32, Batch, Time, DObsCandlesticksBTC] = obs_logical_segments[2]
        obs_history_candlesticks_ftm: Tensor3[float32, Batch, Time, DObsCandlesticksFTM] = obs_logical_segments[3]
        obs_history_santiment_btc_1h: Tensor3[float32, Batch, Time, DObsSantimentBTC1h] = obs_logical_segments[4]
        obs_history_santiment_btc_1d: Tensor3[float32, Batch, Time, DObsSantimentBTC1d] = obs_logical_segments[5]
        obs_history_santiment_ftm_1h: Tensor3[float32, Batch, Time, DObsSantimentFTM1h] = obs_logical_segments[6]
        obs_history_santiment_ftm_1d: Tensor3[float32, Batch, Time, DObsSantimentFTM1d] = obs_logical_segments[7]

        return (
            obs_history_time,
            obs_history_account,
            obs_history_candlesticks_btc,
            obs_history_candlesticks_ftm,
            obs_history_santiment_btc_1h,
            obs_history_santiment_btc_1d,
            obs_history_santiment_ftm_1h,
            obs_history_santiment_ftm_1d
        )
    
class TimeEncoding(Layer):
    def __init__(self, num_obs_in_history: int = 168, d_time_enc: int = 256 // 8, name='time_encoding'):
        super().__init__(name=name)

        self.num_obs_in_history = num_obs_in_history
        self.d_time_enc = d_time_enc

        self.dense_time_encoding = Dense(units=self.d_time_enc, activation=None)
        self.layer_norm_time = LayerNormalization()

    def call(
            self,
            obs_history_time: Tensor3[float32, Batch, Time, DObsTime],
            training: bool
        ) -> Tensor3[float32, Batch, Time, DTimeEnc]:
        
        b_size: Tensor0[int32] = tf.shape(obs_history_time)[0]
        
        time_abs_scaled: Tensor3[float32, Batch, Time, Const1] = tf.tile(
            tf.linspace(start=0., stop=1., num=self.num_obs_in_history)[tf.newaxis, :, tf.newaxis],
            multiples=[b_size, 1, 1]
        )

        time_calendar_clock_scaled: Tensor3[float32, Batch, Time, DObsTime] = obs_history_time

        time_concat: Tensor3[float32, Batch, Time, DTimeConcat] = tf.concat(
            [time_abs_scaled, time_calendar_clock_scaled], axis=-1
        )

        obs_history_time_enc: Tensor3[float32, Batch, Time, DTimeEnc] = self.dense_time_encoding(time_concat)
        obs_history_time_enc: Tensor3[float32, Batch, Time, DTimeEnc] = self.layer_norm_time(obs_history_time_enc, training=training)

        return obs_history_time_enc
    
class AccountEncoding(Layer):
    def __init__(self, d_account_enc: int = 256 // 8, name='account_encoding'):
        super().__init__(name=name)

        self.d_account_enc = d_account_enc

        self.dense_account_encoding = Dense(units=self.d_account_enc, activation=None)
        self.layer_norm_account = LayerNormalization()

    def call(
            self,
            obs_history_account: Tensor3[float32, Batch, Time, DObsAccount],
            training: bool
        ) -> Tensor3[float32, Batch, Time, DAccountEnc]:

        obs_history_account_enc: Tensor3[float32, Batch, Time, DAccountEnc] = self.dense_account_encoding(obs_history_account)
        obs_history_account_enc: Tensor3[float32, Batch, Time, DAccountEnc] = self.layer_norm_account(obs_history_account_enc, training=training)

        return obs_history_account_enc
    
class ObsEncodingInternal(Layer):
    def __init__(
            self,
            num_obs_in_history: int = 168,
            d_time_enc: int = 256 // 8,
            d_account_enc: int = 256 // 8,
            name='observation_encoding_internal'
        ):

        super().__init__(name=name)

        self.num_obs_in_history = num_obs_in_history
        self.d_time_enc = d_time_enc
        self.d_account_enc = d_account_enc

        self.time_encoding = TimeEncoding(num_obs_in_history=self.num_obs_in_history, d_time_enc=self.d_time_enc)
        self.account_encoding = AccountEncoding(d_account_enc=self.d_account_enc)

    def call(
            self,
            obs_history_time: Tensor3[float32, Batch, Time, DObsTime],
            obs_history_account: Tensor3[float32, Batch, Time, DObsAccount],
            training: bool
        ) -> Tensor3[float32, Batch, Time, DObsInternalEnc]:
        
        obs_history_time_enc: Tensor3[float32, Batch, Time, DTimeEnc] = self.time_encoding(
            obs_history_time=obs_history_time,
            training=training
        )
        
        obs_history_account_enc: Tensor3[float32, Batch, Time, DAccountEnc] = self.account_encoding(
            obs_history_account=obs_history_account,
            training=training
        )

        obs_history_internal_enc: Tensor3[float32, Batch, Time, DObsInternalEnc] = tf.concat(
            [obs_history_time_enc, obs_history_account_enc], axis=-1
        )

        return obs_history_internal_enc
    
class CandlesticksEncoding(Layer):
    def __init__(
            self,
            d_candlesticks_btc: int = 60,
            d_candlesticks_ftm: int = 60,
            d_candlesticks_enc: int = 256 * 3 // 8,
            name='candlesticks_encoding'
        ):

        super().__init__(name=name)

        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_ftm = d_candlesticks_ftm
        self.d_candlesticks_enc = d_candlesticks_enc

        self.d_candlesticks_enc_btc = self.d_candlesticks_enc * 4 // 3
        self.d_candlesticks_enc_ftm = self.d_candlesticks_enc * 4 // 3

        self.dense_candlesticks_btc = Dense(units=self.d_candlesticks_enc_btc, activation='gelu')
        self.dense_candlesticks_ftm = Dense(units=self.d_candlesticks_enc_ftm, activation='gelu')

        self.dense_candlesticks_encoding = Dense(units=self.d_candlesticks_enc, activation=None)
        self.layer_norm_candlesticks = LayerNormalization()

    def call(
        self,
        obs_history_candlesticks_btc: Tensor3[float32, Batch, Time, DObsCandlesticksBTC],
        obs_history_candlesticks_ftm: Tensor3[float32, Batch, Time, DObsCandlesticksFTM],
        training: bool
    ) -> Tensor3[float32, Batch, Time, DCandlesticksEnc]:

        obs_history_candlesticks_enc_btc: Tensor3[float32,Batch, Time, DCandlesticksEncBTC] = self.dense_candlesticks_btc(
            obs_history_candlesticks_btc
        )

        obs_history_candlesticks_enc_ftm: Tensor3[float32, Batch, Time, DCandlesticksEncFTM] = self.dense_candlesticks_ftm(
            obs_history_candlesticks_ftm
        )

        obs_history_candlesticks_enc_concat: Tensor3[float32, Batch, Time, DCandlesticksEncConcat] = tf.concat(
            [obs_history_candlesticks_enc_btc, obs_history_candlesticks_enc_ftm], axis=-1
        )

        obs_history_candlesticks_enc: Tensor3[float32, Batch, Time, DCandlesticksEnc] = self.dense_candlesticks_encoding(
            obs_history_candlesticks_enc_concat
        )
        
        obs_history_candlesticks_enc: Tensor3[float32, Batch, Time, DCandlesticksEnc] = self.layer_norm_candlesticks(
            obs_history_candlesticks_enc, training=training
        )

        return obs_history_candlesticks_enc

class SantimentEncoding(Layer):
    def __init__(
            self,
            d_santiment_btc_1h: int = 32,
            d_santiment_btc_1d: int = 26,
            d_santiment_ftm_1h: int = 31,
            d_santiment_ftm_1d: int = 28,
            d_santiment_enc: int = 256 * 3 // 8,
            name='santiment_encoding'
        ):

        super().__init__(name=name)

        self.d_santiment_btc_1h = d_santiment_btc_1h
        self.d_santiment_btc_1d = d_santiment_btc_1d
        self.d_santiment_ftm_1h = d_santiment_ftm_1h
        self.d_santiment_ftm_1d = d_santiment_ftm_1d
        self.d_santiment_enc = d_santiment_enc

        self.d_santiment_enc_btc = self.d_santiment_enc * 4 // 3
        self.d_santiment_enc_ftm = self.d_santiment_enc * 4 // 3

        self.dense_santiment_btc = Dense(units=self.d_santiment_enc_btc, activation='gelu')
        self.dense_santiment_ftm = Dense(units=self.d_santiment_enc_ftm, activation='gelu')

        self.dense_santiment_encoding = Dense(units=self.d_santiment_enc, activation=None)
        self.layer_norm_santiment = LayerNormalization()

    def call(
        self,
        obs_history_santiment_btc_1h: Tensor3[float32, Batch, Time, DObsSantimentBTC1h],
        obs_history_santiment_btc_1d: Tensor3[float32, Batch, Time, DObsSantimentBTC1d],
        obs_history_santiment_ftm_1h: Tensor3[float32, Batch, Time, DObsSantimentFTM1h],
        obs_history_santiment_ftm_1d: Tensor3[float32, Batch, Time, DObsSantimentFTM1d],
        training: bool
    ) -> Tensor3[float32, Batch, Time, DSantimentEnc]:
        
        obs_history_santiment_btc: Tensor3[float32, Batch, Time, DSantimentBTC] = tf.concat(
            [obs_history_santiment_btc_1h, obs_history_santiment_btc_1d], axis=-1
        )

        obs_history_santiment_ftm: Tensor3[float32, Batch, Time, DSantimentFTM] = tf.concat(
            [obs_history_santiment_ftm_1h, obs_history_santiment_ftm_1d], axis=-1
        )

        obs_history_santiment_enc_btc: Tensor3[float32,Batch, Time, DSantimentEncBTC] = self.dense_santiment_btc(
            obs_history_santiment_btc
        )

        obs_history_santiment_enc_ftm: Tensor3[float32, Batch, Time, DSantimentEncFTM] = self.dense_santiment_ftm(
            obs_history_santiment_ftm
        )
        

        obs_history_santiment_enc_concat: Tensor3[float32, Batch, Time, DSantimentEncConcat] = tf.concat(
            [obs_history_santiment_enc_btc, obs_history_santiment_enc_ftm], axis=-1
        )

        obs_history_santiment_enc: Tensor3[float32, Batch, Time, DSantimentEnc] = self.dense_santiment_encoding(
            obs_history_santiment_enc_concat
        )
        
        obs_history_santiment_enc: Tensor3[float32, Batch, Time, DSantimentEnc] = self.layer_norm_santiment(
            obs_history_santiment_enc,
            training=training
        )

        return obs_history_santiment_enc

class ObsEncodingExternal(Layer):
    def __init__(
            self,
            d_candlesticks_btc: int = 60,
            d_candlesticks_ftm: int = 60,
            d_candlesticks_enc: int = 256 * 3 // 8,
            d_santiment_btc_1h: int = 32,
            d_santiment_btc_1d: int = 26,
            d_santiment_ftm_1h: int = 31,
            d_santiment_ftm_1d: int = 28,
            d_santiment_enc: int = 256 * 3 // 8,
            name='observation_encoding_external'
        ):

        super().__init__(name=name)

        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_ftm = d_candlesticks_ftm
        self.d_candlesticks_enc = d_candlesticks_enc

        self.d_santiment_btc_1h = d_santiment_btc_1h
        self.d_santiment_btc_1d = d_santiment_btc_1d

        self.d_santiment_ftm_1h = d_santiment_ftm_1h
        self.d_santiment_ftm_1d = d_santiment_ftm_1d

        self.d_santiment_enc = d_santiment_enc

        self.candlesticks_encoding = CandlesticksEncoding(
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_candlesticks_ftm=self.d_candlesticks_ftm,
            d_candlesticks_enc=self.d_candlesticks_enc
        )

        self.santiment_encoding = SantimentEncoding(
            d_santiment_btc_1h=self.d_santiment_btc_1h,
            d_santiment_btc_1d=self.d_santiment_btc_1d,
            d_santiment_ftm_1h=self.d_santiment_ftm_1h,
            d_santiment_ftm_1d=self.d_santiment_ftm_1d,
            d_santiment_enc=self.d_santiment_enc
        )

    def call(
        self,
        obs_history_candlesticks_btc: Tensor3[float32, Batch, Time, DObsCandlesticksBTC],
        obs_history_candlesticks_ftm: Tensor3[float32, Batch, Time, DObsCandlesticksFTM],
        obs_history_santiment_btc_1h: Tensor3[float32, Batch, Time, DObsSantimentBTC1h],
        obs_history_santiment_btc_1d: Tensor3[float32, Batch, Time, DObsSantimentBTC1d],
        obs_history_santiment_ftm_1h: Tensor3[float32, Batch, Time, DObsSantimentFTM1h],
        obs_history_santiment_ftm_1d: Tensor3[float32, Batch, Time, DObsSantimentFTM1d],
        training: bool
    ) -> Tensor3[float32, Batch, Time, DObsExternalEnc]:
        
        obs_history_candlesticks_enc: Tensor3[float32, Batch, Time, DCandlesticksEnc] = self.candlesticks_encoding(
            obs_history_candlesticks_btc=obs_history_candlesticks_btc,
            obs_history_candlesticks_ftm=obs_history_candlesticks_ftm,
            training=training
        )
        
        obs_history_santiment_enc: Tensor3[float32, Batch, Time, DSantimentEnc] = self.santiment_encoding(
            obs_history_santiment_btc_1h=obs_history_santiment_btc_1h,
            obs_history_santiment_btc_1d=obs_history_santiment_btc_1d,
            obs_history_santiment_ftm_1h=obs_history_santiment_ftm_1h,
            obs_history_santiment_ftm_1d=obs_history_santiment_ftm_1d,
            training=training
        )

        obs_external_enc: Tensor3[float32, Batch, Time, DObsExternalEnc] = tf.concat(
            [obs_history_candlesticks_enc, obs_history_santiment_enc], axis=-1
        )

        return obs_external_enc
    
class Stem(Layer):
    def __init__(
            self,
            num_obs_in_history: int = 168,
            d_time_enc: int = 256 // 8,
            d_account_enc: int = 256 // 8,
            d_candlesticks_btc: int = 60,
            d_candlesticks_ftm: int = 60,
            d_candlesticks_enc: int = 256 * 3 // 8,
            d_santiment_btc_1h: int = 32,
            d_santiment_btc_1d: int = 26,
            d_santiment_ftm_1h: int = 31,
            d_santiment_ftm_1d: int = 28,
            d_santiment_enc: int = 256 * 3 // 8,
            name='stem'
        ):

        super().__init__(name=name)

        self.num_obs_in_history = num_obs_in_history
        self.d_time_enc = d_time_enc
        self.d_account_enc = d_account_enc

        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_ftm = d_candlesticks_ftm
        self.d_candlesticks_enc = d_candlesticks_enc
        self.d_santiment_btc_1h = d_santiment_btc_1h
        self.d_santiment_btc_1d = d_santiment_btc_1d
        self.d_santiment_ftm_1h = d_santiment_ftm_1h
        self.d_santiment_ftm_1d = d_santiment_ftm_1d
        self.d_santiment_enc = d_santiment_enc

        self.obs_encoding_internal = ObsEncodingInternal(
            num_obs_in_history=self.num_obs_in_history,
            d_time_enc=self.d_time_enc,
            d_account_enc=self.d_account_enc
        )
        
        self.obs_encoding_external = ObsEncodingExternal(
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_candlesticks_ftm=self.d_candlesticks_ftm,
            d_candlesticks_enc=self.d_candlesticks_enc,
            d_santiment_btc_1h=self.d_santiment_btc_1h,
            d_santiment_btc_1d=self.d_santiment_btc_1d,
            d_santiment_ftm_1h=self.d_santiment_ftm_1h,
            d_santiment_ftm_1d=self.d_santiment_ftm_1d,
            d_santiment_enc=self.d_santiment_enc
        )

    def call(
            self,
            obs_history_time: Tensor3[float32, Batch, Time, DObsTime],
            obs_history_account: Tensor3[float32, Batch, Time, DObsAccount],
            obs_history_candlesticks_btc: Tensor3[float32, Batch, Time, DObsCandlesticksBTC],
            obs_history_candlesticks_ftm: Tensor3[float32, Batch, Time, DObsCandlesticksFTM],
            obs_history_santiment_btc_1h: Tensor3[float32, Batch, Time, DObsSantimentBTC1h],
            obs_history_santiment_btc_1d: Tensor3[float32, Batch, Time, DObsSantimentBTC1d],
            obs_history_santiment_ftm_1h: Tensor3[float32, Batch, Time, DObsSantimentFTM1h],
            obs_history_santiment_ftm_1d: Tensor3[float32, Batch, Time, DObsSantimentFTM1d],
            training: bool
        ) -> Tuple[
            Tensor3[float32, Batch, Time, DObsInternalEnc],
            Tensor3[float32, Batch, Time, DObsExternalEnc]
        ]:

        obs_history_internal_enc: Tensor3[float32, Batch, Time, DObsInternalEnc] = self.obs_encoding_internal(
            obs_history_time=obs_history_time,
            obs_history_account=obs_history_account,
            training=training
        )

        obs_history_external_enc: Tensor3[float32, Batch, Time, DObsExternalEnc] = self.obs_encoding_external(
            obs_history_candlesticks_btc=obs_history_candlesticks_btc,
            obs_history_candlesticks_ftm=obs_history_candlesticks_ftm,
            obs_history_santiment_btc_1h=obs_history_santiment_btc_1h,
            obs_history_santiment_btc_1d=obs_history_santiment_btc_1d,
            obs_history_santiment_ftm_1h=obs_history_santiment_ftm_1h,
            obs_history_santiment_ftm_1d=obs_history_santiment_ftm_1d,
            training=training
        )

        return obs_history_internal_enc, obs_history_external_enc
    
class StemOutputConcatenation(Layer):
    def __init__(self, name='stem_output_concatenation'):
        super().__init__(name=name)

    def call(
            self,
            obs_history_internal_enc: Tensor3[float32, Batch, Time, DObsInternalEnc],
            obs_history_external_enc: Tensor3[float32, Batch, Time, DObsExternalEnc]
        ) -> Tensor3[float32, Batch, Time, DObsStemEnc]:

        obs_history_stem_enc: Tensor3[float32, Batch, Time, DObsStemEnc] = tf.concat(
            [obs_history_internal_enc, obs_history_external_enc], axis=-1
        )

        return obs_history_stem_enc
    
class AttentionBlock(Layer):
    def __init__(
            self,
            num_heads: int = 4,
            dropout_rate: float = 0.1,
            d_obs_internal_enc: int = 256 // 4,
            d_obs_external_enc: int = 256 * 3 // 4,
            name='attention_block'
        ):

        super().__init__(name=name)

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.d_obs_internal_enc = d_obs_internal_enc
        self.d_obs_external_enc = d_obs_external_enc

        self.d_model = self.d_obs_internal_enc + self.d_obs_external_enc
        self.d_key = self.d_model // num_heads
        self.d_ff_intermediate = self.d_model * 3 // 2

        self.attn_mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_key)
        self.attn_dense = Dense(units=self.d_obs_external_enc, activation=None)
        self.attn_add = Add()
        self.attn_layer_norm = LayerNormalization()

        self.ff_dense_1 = Dense(units=self.d_ff_intermediate, activation='gelu')
        self.ff_dense_2 =  Dense(units=self.d_obs_external_enc, activation=None)
        self.ff_dropout =  Dropout(rate=self.dropout_rate)
        self.ff_add = Add()
        self.ff_layer_norm = LayerNormalization()
  
    def call(
            self,
            obs_history_internal_enc: Tensor3[float32, Batch, Time, DObsInternalEnc],
            obs_history_external_enc: Tensor3[float32, Batch, Time, DObsExternalEnc],
            training: bool
        ) -> Tensor3[float32, Batch, Time, DObsExternalEnc]:

        attn_block_input: Tensor3[float32, Batch, Time, DObsExternalEnc] = obs_history_external_enc     
        obs_history_enc: Tensor3[float32, Batch, Time, DObsStemEnc] = tf.concat([obs_history_internal_enc, attn_block_input], axis=-1)
        attn_mha_output: Tensor3[float32, Batch, Time, DObsStemEnc] = self.attn_mha(query=obs_history_enc, value=obs_history_enc, training=training)
        attn_dense_output: Tensor3[float32, Batch, Time, DObsExternalEnc] = self.attn_dense(attn_mha_output)
        attn_add_output: Tensor3[float32, Batch, Time, DObsExternalEnc] = self.attn_add([attn_block_input, attn_dense_output])
        attn_block_output: Tensor3[float32, Batch, Time, DObsExternalEnc] = self.attn_layer_norm(attn_add_output, training=training)

        ff_block_input: Tensor3[float32, Batch, Time, DObsExternalEnc] = attn_block_output
        obs_history_enc: Tensor3[float32, Batch, Time, DObsStemEnc] = tf.concat([obs_history_internal_enc, ff_block_input], axis=-1)
        ff_dense_1_output: Tensor3[float32, Batch, Time, DFeedForwardInter] = self.ff_dense_1(obs_history_enc)
        ff_dense_2_output: Tensor3[float32, Batch, Time, DObsExternalEnc] = self.ff_dense_2(ff_dense_1_output)
        ff_dropout_output: Tensor3[float32, Batch, Time, DObsExternalEnc] = self.ff_dropout(ff_dense_2_output, training=training)
        ff_add_output = self.ff_add([ff_block_input, ff_dropout_output])
        ff_block_output = self.ff_layer_norm(ff_add_output, training=training)
    
        return ff_block_output

class TransformerOutputTimePooling(Layer):
    def __init__(self, name='output_pooling'):
        super().__init__(name=name)

    def call(
            self,
            obs_history_stem_enc: Tensor3[float32, Batch, Time, DObsStemEnc],
            obs_history_transformer_enc: Tensor3[float32, Batch, Time, DObsExternalEnc]
        ) -> Tensor2[float32, Batch, DObsExternalEnc]:

        last_obs_stem_enc: Tensor2[float32, Batch, DObsStemEnc] = obs_history_stem_enc[:, -1, :]
        last_obs_transformer_enc: Tensor2[float32, Batch, DObsExternalEnc] = obs_history_transformer_enc[:, -1, :]

        last_obs_enc: Tensor2[float32, Batch, DObsTranfEnc] = tf.concat(
            [last_obs_stem_enc, last_obs_transformer_enc], axis=-1
        )

        return last_obs_enc

class ActionBranch(Layer):
    def __init__(self, num_outputs: int, name: str = 'action_branch'):
        super().__init__(name=name)

        self.num_outputs = num_outputs
        self.dense = Dense(units=num_outputs, activation=None)

    def call(
            self,
            inputs: Tensor2[float32, Batch, DObsTranfEnc]
        ) -> Tensor2[float32, Batch, NOutputs]:

        act_logits: Tensor2[float32, Batch, NOutputs] = self.dense(inputs)

        return act_logits

class ValueBranch(Layer):
    def __init__(self, name: str = 'value_branch'):
        super().__init__(name=name)

        self.dense = Dense(units=1, activation=None)

    def call(
            self,
            inputs: Tensor2[float32, Batch, DObsTranfEnc]
        ) -> Tensor2[float32, Batch, Const1]:

        value: Tensor2[float32, Batch, Const1] = self.dense(inputs)

        return value

class Transformer(Layer):
    def __init__(
        self,
        d_history_flat: int = 168 * 181,
        num_obs_in_history: int = 168,
        d_obs: int = 181,
        d_time: int = 2,
        d_account: int = 2,
        d_candlesticks_btc: int = 60,
        d_candlesticks_ftm: int = 60,
        d_santiment_btc_1h: int = 32,
        d_santiment_btc_1d: int = 26,
        d_santiment_ftm_1h: int = 31,
        d_santiment_ftm_1d: int = 28,
        d_obs_enc: int = 256,
        num_attn_blocks: int = 3,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        num_outputs: int = 4,
        name='transformer'
    ):
        
        super().__init__(name=name) 

        self.d_history_flat = d_history_flat
        self.num_obs_in_history = num_obs_in_history
        self.d_obs = d_obs

        self.d_time = d_time
        self.d_account = d_account
        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_ftm = d_candlesticks_ftm
        self.d_santiment_btc_1h = d_santiment_btc_1h
        self.d_santiment_btc_1d = d_santiment_btc_1d
        self.d_santiment_ftm_1h = d_santiment_ftm_1h
        self.d_santiment_ftm_1d = d_santiment_ftm_1d

        self.d_obs_enc = d_obs_enc
        self.num_attn_blocks = num_attn_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.num_outputs = num_outputs

        self.d_time_enc = self.d_obs_enc // 8
        self.d_account_enc = self.d_obs_enc // 8
        self.d_candlesticks_enc = self.d_obs_enc * 3 // 8
        self.d_santiment_enc = self.d_obs_enc * 3 // 8

        self.d_obs_internal_enc = self.d_time_enc + self.d_account_enc
        self.d_obs_external_enc = self.d_candlesticks_enc + self.d_santiment_enc

        self.d_obs_logical_segments = [
            self.d_time,
            self.d_account,
            self.d_candlesticks_btc,
            self.d_candlesticks_ftm,
            self.d_santiment_btc_1h,
            self.d_santiment_btc_1d,
            self.d_santiment_ftm_1h,
            self.d_santiment_ftm_1d
        ]

        assert self.d_obs_enc % 8 == 0
        assert self.d_history_flat == self.num_obs_in_history * self.d_obs
        assert self.d_obs == sum(self.d_obs_logical_segments)

        self.input_split = InputSplit(
            num_obs_in_history=self.num_obs_in_history,
            d_obs=self.d_obs,
            d_obs_logical_segments=self.d_obs_logical_segments
        )

        self.stem = Stem(
            num_obs_in_history=self.num_obs_in_history,
            d_time_enc=self.d_time_enc,
            d_account_enc=self.d_account_enc,
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_candlesticks_ftm=self.d_candlesticks_ftm,
            d_candlesticks_enc=self.d_candlesticks_enc,
            d_santiment_btc_1h=self.d_santiment_btc_1h,
            d_santiment_btc_1d=self.d_santiment_btc_1d,
            d_santiment_ftm_1h=self.d_santiment_ftm_1h,
            d_santiment_ftm_1d=self.d_santiment_ftm_1d,
            d_santiment_enc=self.d_santiment_enc
        )

        self.stem_output_concatenation = StemOutputConcatenation()

        self.attention_blocks: List[AttentionBlock] = [
            AttentionBlock(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                d_obs_internal_enc=self.d_obs_internal_enc,
                d_obs_external_enc=self.d_obs_external_enc,
                name=f'attention_block_{i}'
            ) for i in range(self.num_attn_blocks)
        ]

        self.transformer_output_time_pooling = TransformerOutputTimePooling()

        self.action_branch = ActionBranch(num_outputs=self.num_outputs)
        self.value_branch = ValueBranch()

    def call(self, obs_history_flat: Tensor2[float32, Batch, DFlatHistory], training: bool):
        
        obs_history_flat: Tensor2[float32, Batch, DFlatHistory] = obs_history_flat

        obs_logical_segments: Tuple[Tensor3] = self.input_split(obs_history_flat=obs_history_flat)
        
        obs_history_time: Tensor3[float32, Batch, Time, DObsTime] = obs_logical_segments[0]
        obs_history_account: Tensor3[float32, Batch, Time, DObsAccount] = obs_logical_segments[1]
        obs_history_candlesticks_btc: Tensor3[float32, Batch, Time, DObsCandlesticksBTC] = obs_logical_segments[2]
        obs_history_candlesticks_ftm: Tensor3[float32, Batch, Time, DObsCandlesticksFTM] = obs_logical_segments[3]
        obs_history_santiment_btc_1h: Tensor3[float32, Batch, Time, DObsSantimentBTC1h] = obs_logical_segments[4]
        obs_history_santiment_btc_1d: Tensor3[float32, Batch, Time, DObsSantimentBTC1d] = obs_logical_segments[5]
        obs_history_santiment_ftm_1h: Tensor3[float32, Batch, Time, DObsSantimentFTM1h] = obs_logical_segments[6]
        obs_history_santiment_ftm_1d: Tensor3[float32, Batch, Time, DObsSantimentFTM1d] = obs_logical_segments[7]

        obs_logical_segments_enc: Tuple[Tensor3] = self.stem(
            obs_history_time=obs_history_time,
            obs_history_account=obs_history_account,
            obs_history_candlesticks_btc=obs_history_candlesticks_btc,
            obs_history_candlesticks_ftm=obs_history_candlesticks_ftm,
            obs_history_santiment_btc_1h=obs_history_santiment_btc_1h,
            obs_history_santiment_btc_1d=obs_history_santiment_btc_1d,
            obs_history_santiment_ftm_1h=obs_history_santiment_ftm_1h,
            obs_history_santiment_ftm_1d=obs_history_santiment_ftm_1d,
            training=training
        )
        
        obs_history_internal_enc: Tensor3[float32, Batch, Time, DObsInternalEnc] = obs_logical_segments_enc[0]
        obs_history_external_enc: Tensor3[float32, Batch, Time, DObsExternalEnc] = obs_logical_segments_enc[1]

        obs_history_stem_enc: Tensor3[float32, Batch, Time, DObsStemEnc] = self.stem_output_concatenation(
            obs_history_internal_enc=obs_history_internal_enc,
            obs_history_external_enc=obs_history_external_enc
        )

        for i in range(self.num_attn_blocks):
            obs_history_external_enc: Tensor3[float32, Batch, Time, DObsExternalEnc] = self.attention_blocks[i](
                obs_history_internal_enc=obs_history_internal_enc,
                obs_history_external_enc=obs_history_external_enc,
                training=training
            )

        obs_history_transformer_enc: Tensor3[float32, Batch, Time, DObsExternalEnc] = obs_history_external_enc

        current_obs_transformer_enc: Tensor2[float32, Batch, DObsTranfEnc] = self.transformer_output_time_pooling(
            obs_history_stem_enc=obs_history_stem_enc,
            obs_history_transformer_enc=obs_history_transformer_enc
        )

        logits: Tensor2[float32, Batch, NOutputs] = self.action_branch(inputs=current_obs_transformer_enc)
        values: Tensor2[float32, Batch, Const1] = self.value_branch(inputs=current_obs_transformer_enc)

        return logits, values

class TransformerModel(Model):
    def __init__(
        self,
        d_history_flat: int = 168 * 181,
        num_obs_in_history: int = 168,
        d_obs: int = 181,
        d_time: int = 2,
        d_account: int = 2,
        d_candlesticks_btc: int = 60,
        d_candlesticks_ftm: int = 60,
        d_santiment_btc_1h: int = 32,
        d_santiment_btc_1d: int = 26,
        d_santiment_ftm_1h: int = 31,
        d_santiment_ftm_1d: int = 28,
        d_obs_enc: int = 256,
        num_attn_blocks: int = 3,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        num_outputs: int = 4
    ):
        
        self.d_history_flat = d_history_flat
        self.num_obs_in_history = num_obs_in_history
        self.d_obs = d_obs
        self.d_time = d_time
        self.d_account = d_account
        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_ftm = d_candlesticks_ftm
        self.d_santiment_btc_1h = d_santiment_btc_1h
        self.d_santiment_btc_1d = d_santiment_btc_1d
        self.d_santiment_ftm_1h = d_santiment_ftm_1h
        self.d_santiment_ftm_1d = d_santiment_ftm_1d
        self.d_obs_enc = d_obs_enc

        self.num_attn_blocks = num_attn_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.num_outputs = num_outputs

        inputs: Tensor2[float32, Batch, DFlatHistory] = Input(shape=(self.d_history_flat,), name='inputs')
        
        transformer_layer = Transformer(
            d_history_flat=self.d_history_flat,
            num_obs_in_history=self.num_obs_in_history,
            d_obs=self.d_obs,
            d_time=self.d_time,
            d_account=self.d_account,
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_candlesticks_ftm=self.d_candlesticks_ftm,
            d_santiment_btc_1h=self.d_santiment_btc_1h,
            d_santiment_btc_1d=self.d_santiment_btc_1d,
            d_santiment_ftm_1h=self.d_santiment_ftm_1h,
            d_santiment_ftm_1d=self.d_santiment_ftm_1d,
            d_obs_enc=self.d_obs_enc,
            num_attn_blocks=self.num_attn_blocks,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            num_outputs=self.num_outputs
        )

        logits: Tensor2[float32, Batch, NOutputs]
        values: Tensor2[float32, Batch, Const1]
        
        logits, values = transformer_layer(obs_history_flat=inputs, training=True)

        super().__init__(
            inputs=[inputs],
            outputs=[logits, values]
        )

@DeveloperAPI
class TransformerModelAdapter(TFModelV2):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        d_history_flat: int = 168 * 181,
        num_obs_in_history: int = 168,
        d_obs: int = 181,
        d_time: int = 2,
        d_account: int = 2,
        d_candlesticks_btc: int = 60,
        d_candlesticks_ftm: int = 60,
        d_santiment_btc_1h: int = 32,
        d_santiment_btc_1d: int = 26,
        d_santiment_ftm_1h: int = 31,
        d_santiment_ftm_1d: int = 28,
        d_obs_enc: int = 256,
        num_attn_blocks: int = 3,
        num_heads: int = 4,
        dropout_rate: float = 0.1
    ):
        super(TransformerModelAdapter, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        
        self.d_history_flat = d_history_flat
        self.num_obs_in_history = num_obs_in_history
        self.d_obs = d_obs
        self.d_time = d_time
        self.d_account = d_account
        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_ftm = d_candlesticks_ftm
        self.d_santiment_btc_1h = d_santiment_btc_1h
        self.d_santiment_btc_1d = d_santiment_btc_1d
        self.d_santiment_ftm_1h = d_santiment_ftm_1h
        self.d_santiment_ftm_1d = d_santiment_ftm_1d
        self.d_obs_enc = d_obs_enc

        self.num_attn_blocks = num_attn_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.num_outputs = num_outputs

        self.model = TransformerModel(
            d_history_flat=self.d_history_flat,
            num_obs_in_history=self.num_obs_in_history,
            d_obs=self.d_obs,
            d_time=self.d_time,
            d_account=self.d_account,
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_candlesticks_ftm=self.d_candlesticks_ftm,
            d_santiment_btc_1h=self.d_santiment_btc_1h,
            d_santiment_btc_1d=self.d_santiment_btc_1d,
            d_santiment_ftm_1h=self.d_santiment_ftm_1h,
            d_santiment_ftm_1d=self.d_santiment_ftm_1d,
            d_obs_enc=self.d_obs_enc,
            num_attn_blocks=self.num_attn_blocks,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            num_outputs=self.num_outputs
        )

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[Tensor2[float32, Batch, Depth]],
        seq_lens: Tensor1[int32, Batch],
    ) -> Tuple[
        Tensor2[float32, Batch, NOutputs],
        List[Tensor2[float32, Batch, Depth]]
    ]:
                
        obs_history_flat: Tensor2[float32, Batch, DFlatHistory] = input_dict["obs"]
        training: bool = input_dict["is_training"]
        
        logits: Tensor2[float32, Batch, NOutputs]
        values: Tensor2[float32, Batch, Const1]

        logits, values = self.model(inputs=obs_history_flat, training=training)

        self._value_out: Tensor2[float32, Batch, Const1]
        self._value_out = values

        return logits, []
    
    @override(ModelV2)
    def value_function(self) -> Tensor1[float32, Batch]:
        return tf.reshape(self._value_out, [-1])
    

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    transformer_model = TransformerModel()
    transformer_model.summary()
    
    obs_history_flat = tf.random.uniform(shape=[15, 168 * 183], minval=0, maxval=10)
    logits, values = transformer_model(inputs=obs_history_flat)
    
    print(values)