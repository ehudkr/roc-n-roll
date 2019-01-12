from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from bokeh.plotting import output_file, save
from bokeh.layouts import gridplot, row

from rocnroll import plot_roc_curve, plot_pr_curve


if __name__ == '__main__':

    X, y = make_classification(n_samples=500,
                               n_features=20, n_informative=2, n_redundant=15,
                               n_classes=2, flip_y=0.1,
                               class_sep=0.5,
                               weights=[0.8, 0.2],
                               random_state=2018)
    clf = LogisticRegression()
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]

    p_roc = plot_roc_curve(y, y_pred, binary_metric_funcs={"accuracy": metrics.accuracy_score,
                                                           "f1": metrics.f1_score,
                                                           "conf_matrix": metrics.confusion_matrix})
    p_pr = plot_pr_curve(y, y_pred, binary_metric_funcs={"accuracy": metrics.accuracy_score,
                                                         "log_loss": metrics.log_loss})
    p = gridplot([[p_roc, p_pr]])
    # p = row(p_roc, p_pr)
    output_file("example.html")
    save(p)
