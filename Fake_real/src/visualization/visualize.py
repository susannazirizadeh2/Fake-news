import seaborn as sns
import matplotlib.pyplot as plt



def conf_heat(cf_matrix,):
	ax = sns.heatmap(cf_matrix, annot=True, annot_kws={"size": 16})
	ax.set_ylim([0,2]);

	ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
	ax.set_title('Confusion Matrix'); 
	ax.xaxis.set_ticklabels(['Fake', 'Real']); ax.yaxis.set_ticklabels(['Fake', 'Real']);
	plt.show()
