#coding: utf-8
import optuna
import torch
import torchvision

from src.losses import MSE_loss, VGG_loss ## （このコードがあるディレクトリ）/src/losses.pyからMSE_loss、VGG_lossをimport
                                          ## このダミーコードそのままではもちろん動きません
from src.models import Model1
from src.dataset import train_dataset


## まずは関数の設定

def main_training(a, b, optimizer):
	
	## 通常の学習コード部分（に当たるもの）.「既存コードにある」と想定.
	## この例ではepoch_loss（データセットで得られるロスの平均値）を
	## optunaの目的関数にする.
	## a がMSE_lossの重み係数、bがVGG_lossの重み係数

	epoch_loss = 0
	model.train()

	for iteration, mini_batch in enumerate(train_data_loader, 1):
		input_images, ground_truth = mini_batch[0], mini_batch[1] ## 各イテレーションでdataloaderから引っ張ってくるものがmini_batch.
		                                                          ## 各mini_batch自体は[input_images, （対応する）ground_truth]で構成されている.
		optimizer.zero_grad()

		output = model(input_images)

		mse_loss = MSE_loss(output, ground_truth)

		vgg_loss = VGG_loss(output, ground_truth)

		total_loss = a*mse_loss + b*vgg_loss

		total_loss.backward()
		optimizer.step()

		epoch_loss += total_loss.data

	epoch_loss /= len(train_data_loader)
	
	return epoch_loss 


## *******************************************************************************************************************************************************************************
def objective(trail):
	## f(a, b) = a*MSE_loss + b*VGG_lossを最小にするaとb、および良い最適化手法を探索する。

	## 最適化させたい（探索したい）パラメータの設定
	a = trial.suggest_uniform('a', 1, 10)                               ## MSE_lossの重み係数は（最初は）[1, 10)の区間で一様な確率でサンプリングした値を使う。
	b = trial.suggest_log_uniform('b', 1e-5, 1e-1)                      ## VGG_lossの重み係数は（最初は）[1e-5, 1e-1)の区間で対数一様分布でサンプリングした値を使う。
	optimizer_label = trial.suggest_categorical('optimizer', ['SGD', 'Adam']) ## 最適化手法はSGD, Adamのどちらか
	if optimizer_label == 'SGD':
		optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=0.9)
	elif optimizer_label == 'Adam':
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)

	epoch_loss = main_training(a, b, optimizer)  ## 目的関数の設定（既存コードに追加で付け加えるならこれだけで良い）
	return epoch_loss
## *******************************************************************************************************************************************************************************








## loss, model, train_dataset(train_data_loader)の設定。ダミーコードなのでここは適当です。
MSE_loss = MSE_loss()
VGG_loss = VGG_loss()

model = Model1()

train_dataset = train_dataset(dataset_dir='./data/aaaaaaaaaa')
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=8, batch_size=32, shuffle=True)



## *********************************************************************************************************************************
study = optuna.create_study(study_name='optim_mse_and_vgg_weight_and_optimizer',
	                        storage='sqlite:///optuna_database_dir/mse_vgg_optimizer.db',
	                        load_if_exists=True,
	                        pruner=optuna.pruners.MedianPruner(n_startup_trails=5, n_warmup_steps=20))
                            ## この設定は
                            ## “最初の5通りは枝刈りをせず、6通り目以降の探索でも、20エポックは枝刈りを実施しない（で見守る）、
                            ## 21エポック以降は毎回「枝刈りするかしないか」をチェックする”

study.optimize(objective, n_trials=100) ## 100通りのパターン探索
## **********************************************************************************************************************************

print('best_coef_combinations:\n')
print(study.best_params) ## 最適なパラメータ（a, b, optimizer）を出力
print('\n\nbest_loss:\n')
print(study.best_value)  ## 最適パラメータでの目的関数の値
print('\n')

## 以下おまけ
df = study.trials_dataframe()
df.to_csv('./csv_result/mse_vgg_optimizer.csv')