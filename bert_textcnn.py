from keras_bert import load_trained_model_from_checkpoint,Tokenizer
from tensorflow import keras
import codecs
from tensorflow.keras.layers import *



def textcnn(inputs):
	print(inputs)
	# 3,4,5
	cnn1 = keras.layers.Conv1D(
			256,
			3,
			strides=1,
			padding='same',
			activation='relu'
		)(inputs) # shape=[batch_size,maxlen-2,256]
	cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]

	cnn2 = keras.layers.Conv1D(
			256,
			4,
			strides=1,
			padding='same',
			activation='relu'

		)(inputs)
	cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

	cnn3 = keras.layers.Conv1D(
			256,
			5,
			strides=1,
			padding='same'
		)(inputs)
	cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

	output = keras.layers.concatenate(
		[cnn1,cnn2,cnn3],
		axis=-1)
	output = keras.layers.Dropout(0.2)(output)
	return output

def build_bert_model(config_path,checkpoint_path,class_nums):
	# bert = build_transformer_model(
	# 	config_path=config_path,
	# 	checkpoint_path=checkpoint_path,
	# 	model='bert',
	# 	return_keras_model=False)

	bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

	for layer in bert_model.layers:
		layer.tainable = True


	x1_in = Input(shape=(None,))
	x2_in = Input(shape=(None,))

	x = bert_model([x1_in, x2_in])

	cls_features = Lambda(lambda x: x[:, 0])(x)

	all_token_embedding = Lambda(
		lambda x:x[:,1:-1],
		name='all-token'
		)(x) #shape=[batch_size,maxlen-2,768]

	cnn_features = textcnn(
		all_token_embedding) #shape=[batch_size,cnn_output_dim]
	concat_features = keras.layers.concatenate(
		[cls_features,cnn_features],
		axis=-1)

	dense = keras.layers.Dense(
			units=512,
			activation='relu'
		)(concat_features)

	output = keras.layers.Dense(
			units=class_nums,
			activation='softmax'
		)(dense)

	model = keras.models.Model([x1_in, x2_in],output)

	return model


if __name__ == '__main__':
    filepath = "/home/lzh/Documents/chinese_wwm_ext/"
    config_path = filepath+"bert_config.json"
    checkpoint_path=filepath+"bert_model.ckpt"
    class_nums=13
    build_bert_model(config_path, checkpoint_path, class_nums)