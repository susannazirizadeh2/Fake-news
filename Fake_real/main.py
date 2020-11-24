from src.data.make_dataset import prepare_data, train_test_prepare
from src.features.build_features import make_features
from src.visualization.visualize import conf_heat
from sklearn.metrics import confusion_matrix, classification_report

df = prepare_data()

X_train, X_test, y_train, y_test = train_test_prepare(df)

pipe = make_features()

pipe.fit(X_train, y_train)
print("model score: %.3f" % pipe.score(X_test, y_test))

y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
cf_matrix = confusion_matrix(y_test, y_pred)


conf_heat(cf_matrix)





