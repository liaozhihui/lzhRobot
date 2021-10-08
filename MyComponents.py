import typing
import codecs
from typing import Any, Optional, Text, Dict, List, Type
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.components import Component
from keras_bert import Tokenizer
from tensorflow.keras.optimizers import Adam
import numpy as np
from bert_textcnn import build_bert_model
from tensorflow import keras
filepath = "/home/lzh/Documents/chinese_wwm_ext/"
config_path = filepath+"bert_config.json"
checkpoint_path=filepath+"bert_model.ckpt"
dict_path = filepath+'/vocab.txt'
bast_model_filepath = './checkpoint/best_model.weights.h5'

def seq_padding(X,padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([np.concatenate([x,[padding]*(ML-len(x))]) if len(x)< ML else x for x in X])

class OurTokenizer(Tokenizer):

    def _tokenize(self, text):

        R = []

        for c in text:

            if c in self._token_dict:

                R.append(c)

            elif self._is_space(c):

                R.append( '[unused1]') # space类用未经训练的[unused1]表示

            else:

                R.append( '[UNK]') # 剩余的字符是[UNK]

        return R
token_dict = {}

with codecs.open(dict_path,"r","utf8") as reader:
    for line in reader:
        token=line.strip()
        token_dict[token] = len(token_dict)

tokenizer = OurTokenizer(token_dict)

maxlen = 128

class data_generator:

	def __init__(self, data, batch_size=32):
		self.data = data
		self.batch_size = batch_size
		self.steps = len(self.data) // self.batch_size
		if len(self.data) % self.batch_size != 0:
			self.steps += 1
	def __len__(self):
		return self.steps
	def __iter__(self):
		while True:
			idxs = list(range(len(self.data)))
			np.random.shuffle(idxs)
			X1, X2, Y = [], [], []
			for i in idxs:
				d = self.data[i]
				text = d[0][:maxlen]
				x1, x2 = tokenizer.encode(first=text)
				y = d[1]
				X1.append(x1)
				X2.append(x2)
				Y.append([y])
				if len(X1) == self.batch_size or i == idxs[-1]:
					X1 = seq_padding(X1)
					X2 = seq_padding(X2)
					Y = seq_padding(Y)
					yield [X1, X2], Y
					[X1, X2, Y] = [], [], []


class SentimentAnalyzerWithBertTextCNN(Component):
    """一个预训练的情感识别组建"""

    name = "sentiment"
    provides = ["entities"]
    requires = []
    defaults = {}
    language_list = ["zh"]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

        pass

    def train(
            self,
            training_data: TrainingData,
            config: Optional[RasaNLUModelConfig] = None,
            **kwargs: Any,
    ) -> None:
        """训练阶段代码"""
        label_list = training_data.intents
        label2id = {label: idx for idx, label in enumerate(label_list)}
        data = []
        for temp in training_data.intent_examples:
            data.append((temp.data["text"],label2id.get(temp.data["intent"])))

        random_order = list(range(len(data)))
        np.random.shuffle(random_order)
        train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
        valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

        train_D = data_generator(train_data)
        valid_D = data_generator(valid_data)

        model = build_bert_model(config_path, checkpoint_path, len(label_list))
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(5e-6),
            metrics=['accuracy'],
        )

        earlystop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=2,
            mode='min'
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            bast_model_filepath,
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            mode='auto',
            period=1,
            save_weights_only=False
        )
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=1,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[earlystop, checkpoint]
        )



    def convert_to_rasa(self, value, confidence):
        """把模型的输出转化为 rasa 能够识别的输出."""

        entity = {"value": value,
                  "confidence": confidence,
                  "entity": "sentiment",
                  "extractor": "sentiment_extractor"}

        return entity

    def process(self, message: Message, **kwargs: Any) -> None:
        """使用分类器来处理文本，并且转化为 rasa 能够接受的格式"""

        # sid = SentimentIntensityAnalyzer()
        print('##################### 情感识别 ###########################')
        # res = sid.polarity_scores(message.as_dict_nlu()['text'])
        # key, value = max(res.items(), key=lambda x: x[1])
        # entity = self.convert_to_rasa(key, value)
        # message.set("entities", [entity], add_to_output=True)
        x1,x2 = tokenizer.encode(first=message.as_dict_nlu()['text'])
        print(x1,"\n",x2)
        model = build_bert_model(config_path, checkpoint_path, 13)
        model.load_weights(bast_model_filepath)
        res = model.predict(x1,x2)
        print("result:",res)
        # key, value = max(res.items(), key=lambda x: x[1])
        # entity = self.convert_to_rasa(key, value)
        # message.set("entities", [entity], add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """组建持久化的运行代码"""
        pass