import numpy as np
from matplotlib import pyplot as plt

# yyplot 作成関数
def yyplot(y_obs, y_pred):
    yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
    ymin2, ymax2, yrange2 = np.amin(y_obs), np.amax(y_obs), np.ptp(y_obs)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(y_obs, y_pred)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
    plt.xlim(ymin2 - yrange2 * 0.01, ymax2 + yrange2 * 0.01)
    plt.ylim(ymin2 - yrange2 * 0.01, ymax2 + yrange2 * 0.01)
    plt.xlabel('y_observed', fontsize=24)
    plt.ylabel('y_predicted', fontsize=24)
    plt.title('Observed-Predicted Plot', fontsize=24)
    plt.tick_params(labelsize=16)
    plt.show()

    return fig


# yyplot の実行例
np.random.seed(0)
y_obs = np.load('gt_ar.npy')
y_pred = np.load('train_ar.npy')
fig = yyplot(y_obs, y_pred)
