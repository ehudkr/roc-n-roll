"""
ROC curve and Precision-Recall curve interactive plot.
"""
import pandas as pd
from sklearn import metrics

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool

from collections import namedtuple


BINARY_METRIC_FUNCS = {"accuracy": metrics.accuracy_score,
                       "f1": metrics.f1_score,
                       "precision": metrics.precision_score,
                       "recall": metrics.recall_score,
                       "matthews": metrics.matthews_corrcoef,
                       "log_loss": metrics.log_loss,
                       "conf_matrix": metrics.confusion_matrix}

SCORES_METRIC_FUNCS = {"brier": metrics.brier_score_loss,
                       "roc_auc": metrics.roc_auc_score,
                       "avg_precision": metrics.average_precision_score,
                       "prevalence": lambda y_true, y_score, sample_weight=None:
                                     pd.Series(y_true).value_counts(normalize=True).loc[y_true.max()],
                       "N": lambda y_true, y_score, sample_weight=None: y_true.shape[0]}

CONF_MATRIX_COMPONENTS = ["tn", "fp", "fn", "tp"]  # Ordered by the unravelling of a confusion matrix into vector
# Confusion matrix table formatting in hover tooltip:
CONF_MATRIX_FORMATTING = """<div>
                              <table style="border: 1px solid black; border-collapse: collapse">
                                <tr  style="border: 1px solid black;">
                                  <td style="border: 1px solid black;">@tn</td>
                                  <td style="border: 1px solid black;">@fp</td>
                                </tr>
                                <tr  style="border: 1px solid black;">
                                  <td style="border: 1px solid black;">@fn</td>
                                  <td style="border: 1px solid black;">@tp</td>
                                </tr>
                              </table>
                            </div>"""

# Plot defaults:
PLOT_LIMIT_PAD = 0.05
PLOT_WIDTH = 400
PLOT_HEIGHT = 400
PLOT_LINE_WIDTH = 2.0


def plot_roc_curve(y_true, y_prob, sample_weight=None,
                   binary_metric_funcs=None, scores_metric_funcs=None):
    """
    Compute and plot receiver operating characteristic (ROC) curve
     for binary classification tasks.
     An interactive tool showing multiple metrics is present upon
     hovering over the curve.

    Parameters
    ----------
    y_true : array, shape=(n_sample,)
        True binary labels.
        Labels should be either {0, 1} or {-1, 1}.

    y_prob : array, shape=(n_sample,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions
        (as returned by “decision_function” or "predict_proba" on some classifiers).

    sample_weight : array, shape = (n_samples,), optional
        Sample weights

    binary_metric_funcs : dict, optional
        Collection of metrics to evaluate for prediction binarization at each cutoff value.
        Dictionary which keys are metric names and value is a callable
        with the signature `scorer(y_true, y_pred, **kwargs)`.
        If not provided, uses the metric specified in `BINARY_METRIC_FUNCS`.

    scores_metric_funcs : dict, optional
        Collection of metrics to evaluate once on the non-thresholed y_prob prediction.
        Dictionary which keys are metric names and value is a callable
        with the signature `scorer(y_true, y_pred, **kwargs)`.
        If not provided, uses the metric specified in `SCORES_METRIC_FUNCS`.

    Returns
    -------
    bokeh.plotting.figure:
        Bokeh figure object with ROC curve plotted on
        and interactive hover-tool showing evaluations at each point on curve.
    """
    scores = calculate_scores(y_true, y_prob, sample_weight, metrics.roc_curve,
                              binary_metric_funcs, scores_metric_funcs)
    # If 'fpr' and 'tpr' were in binary_metric_funcs remove them to avoid column duplication:
    scores.per_threshold.drop(columns=["fpr", "tpr"], inplace=True, errors="ignore")
    scores.per_threshold.rename(columns={"first_curve_metric": "fpr",
                                         "second_curve_metric": "tpr"}, inplace=True)
    p = _plot_performance_curve(scores.per_threshold,
                                x_axis_metric_name="fpr", y_axis_metric_name="tpr",
                                curve_type="ROC")
    return p


def plot_pr_curve(y_true, y_prob, sample_weight=None,
                  binary_metric_funcs=None, scores_metric_funcs=None):
    """
    Compute and plot precision-recall (PR) curve
     for binary classification tasks.
     An interactive tool showing multiple metrics is present upon
     hovering over the curve.

    Parameters
    ----------
    y_true : array, shape=(n_sample,)
        True binary labels.
        Labels should be either {0, 1} or {-1, 1}.

    y_prob : array, shape=(n_sample,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions
        (as returned by “decision_function” or "predict_proba" on some classifiers).

    sample_weight : array, shape = (n_samples,), optional
        Sample weights

    binary_metric_funcs : dict, optional
        Collection of metrics to evaluate for prediction binarization at each cutoff value.
        Dictionary which keys are metric names and value is a callable
        with the signature `scorer(y_true, y_pred, **kwargs)`.
        If not provided, uses the metric specified in `BINARY_METRIC_FUNCS`.

    scores_metric_funcs : dict, optional
        Collection of metrics to evaluate once on the non-thresholed y_prob prediction.
        Dictionary which keys are metric names and value is a callable
        with the signature `scorer(y_true, y_pred, **kwargs)`.
        If not provided, uses the metric specified in `SCORES_METRIC_FUNCS`.

    Returns
    -------
    bokeh.plotting.figure:
        Bokeh figure object with ROC curve plotted on
        and interactive hover-tool showing evaluations at each point on curve.
    """
    scores = calculate_scores(y_true, y_prob, sample_weight, metrics.precision_recall_curve,
                              binary_metric_funcs, scores_metric_funcs)
    # If 'precision' and 'recall' were in binary_metric_funcs remove them to avoid column duplication:
    scores.per_threshold.drop(columns=["precision", "recall"], inplace=True, errors="ignore")
    scores.per_threshold.rename(columns={"first_curve_metric": "precision",
                                         "second_curve_metric": "recall"}, inplace=True)
    p = _plot_performance_curve(scores.per_threshold,
                                x_axis_metric_name="recall",
                                y_axis_metric_name="precision",
                                curve_type="PR",
                                pos_class_prevalence=scores.on_probs.get("prevalence", None))
    return p


def calculate_scores(y_true, y_prob, sample_weight=None, curve_metric=metrics.roc_curve,
                     binary_metric_funcs=None, scores_metric_funcs=None):
    """
    Evaluates the metrics specified in `binary_binary_metric_funcs` and
     `scores_scores_metric_funcs` over `y_prob` against `y_true`.

    If confusion matrix ("conf_matrix") is provided as metric, it is
     calculated and then unravelled to its 4 basic components (# of true
     positives, # of false positives, etc.) and stored as for different
     columns with names as specified in CONF_MATRIX_COMPONENTS.

    Parameters
    ----------
    y_true : array, shape=(n_sample,)
        True binary labels.
        Labels should be either {0, 1} or {-1, 1}.

    y_prob : array, shape=(n_sample,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions
        (as returned by “decision_function” or "predict_proba" on some classifiers).

    sample_weight : array, shape = (n_samples,), optional
        Sample weights

    curve_metric : callable, default: sklearn.metrics.roc_curve
        Main metric for calculating a curve using `y_true` and `y_prob`
        and `sample_weight`.
        will usually be either `sklearn.metrics.roc_curve` or
        `sklearn.metrics.precision_recall_curve` and should return a 3-tuple
        (metric_1, metric_2, thresholds).

    binary_metric_funcs : dict, optional
        Collection of metrics to evaluate for prediction binarization at each cutoff value.
        Dictionary which keys are metric names and value is a callable
        with the signature `scorer(y_true, y_pred, **kwargs)`.
        If not provided, uses the metric specified in `BINARY_METRIC_FUNCS`.

    scores_metric_funcs : dict, optional
        Collection of metrics to evaluate once on the non-thresholed y_prob prediction.
        Dictionary which keys are metric names and value is a callable
        with the signature `scorer(y_true, y_pred, **kwargs)`.
        If not provided, uses the metric specified in `SCORES_METRIC_FUNCS`.

    Returns
    -------
    namedtuple :
        Scores
        - per_threshold - pd.DataFrame of shape=
                          (n_threshold, len(binary_binary_metric_funcs).
                          Column names are binary_binary_metric_funcs.keys().
                          Holding evaluations of each metric for each
                          binarization of thresholds.
        - on_probs - pd.Series of shape=(len(scores_metric_funcs),).
                     Index names are scores_metric_funcs.keys().
                     Holding metric evaluations of each metric (once) on the
                     scores (non-thresholed) provided in `y_prob`.
    """
    scores_per_thresh = _calc_scores_per_thresh(y_true, y_prob, curve_metric=curve_metric,
                                                sample_weight=sample_weight,
                                                binary_metric_funcs=binary_metric_funcs)
    scores_on_probs = _calc_scores_on_probs(y_true, y_prob, sample_weight,
                                            scores_metric_funcs=scores_metric_funcs)

    Scores = namedtuple("Scores", ["per_threshold", "on_probs"])
    scores = Scores(scores_per_thresh, scores_on_probs)
    return scores


def _calc_scores_per_thresh(y_true, y_prob, curve_metric=metrics.roc_curve,
                            sample_weight=None, binary_metric_funcs=None):
    """
    Evaluates the metrics specified in `binary_binary_metric_funcs` for
     all thresholds (each binarization of `y_prob`).

    If confusion matrix ("conf_matrix") is provided as metric, it is
     calculated and then unravelled to its 4 basic components (# of true
     positives, # of false positives, etc.) and stored as for different
     columns with names as specified in CONF_MATRIX_COMPONENTS.

    Parameters
    ----------
    y_true : array, shape=(n_sample,)
        True binary labels.
        Labels should be either {0, 1} or {-1, 1}.

    y_prob : array, shape=(n_sample,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions
        (as returned by “decision_function” or "predict_proba" on some classifiers).

    sample_weight : array, shape = (n_samples,), optional
        Sample weights

    curve_metric : callable, default: sklearn.metrics.roc_curve
        Main metric for calculating a curve using `y_true` and `y_prob`
        and `sample_weight`.
        will usually be either `sklearn.metrics.roc_curve` or
        `sklearn.metrics.precision_recall_curve` and should return a 3-tuple
        (metric_1, metric_2, thresholds).

    binary_metric_funcs : dict, optional
        Collection of metrics to evaluate for prediction binarization at each cutoff value.
        Dictionary which keys are metric names and value is a callable
        with the signature `scorer(y_true, y_pred, **kwargs)`.
        If not provided, uses the metric specified in `BINARY_METRIC_FUNCS`.

    Returns
    -------
    pd.DataFrame : shape=(n_threshold, len(binary_binary_metric_funcs).
        Column names are binary_binary_metric_funcs.keys().
        Holding evaluations of each metric for each
        binarization of thresholds.

    """
    binary_metric_funcs = binary_metric_funcs or BINARY_METRIC_FUNCS.copy()

    calc_conf_matrix = "conf_matrix" in binary_metric_funcs.keys()
    if calc_conf_matrix:
        binary_metric_funcs.pop("conf_matrix")

    first_curve_metric, second_curve_metric, thresh = curve_metric(y_true, y_prob,
                                                                   sample_weight=sample_weight)
    scores = []
    # Calculate the scores for each possible threshold:
    for i, t in enumerate(thresh):
        cur_y = y_prob >= t
        cur_scores = {"threshold": t,
                      "first_curve_metric": first_curve_metric[i],
                      "second_curve_metric": second_curve_metric[i]}
        # Score:
        for metric_name, metric_func in binary_metric_funcs.items():
            cur_scores[metric_name] = metric_func(y_true, cur_y, sample_weight=sample_weight)

        # Break confusion matrix into 4 components, and add into scores if binary prediction:
        cur_conf_mat = metrics.confusion_matrix(y_true, cur_y, sample_weight=sample_weight)
        if calc_conf_matrix and cur_conf_mat.shape == (2, 2):
            cur_conf_mat = dict(zip(CONF_MATRIX_COMPONENTS, cur_conf_mat.ravel()))  # convert matrix shape to 4 cols
            cur_scores.update(cur_conf_mat)

        scores.append(cur_scores)

    scores = pd.DataFrame.from_records(scores)
    return scores


def _calc_scores_on_probs(y_true, y_prob, sample_weight=None,
                          scores_metric_funcs=None):
    """

    Parameters
    ----------
    y_true : array, shape=(n_sample,)
        True binary labels.
        Labels should be either {0, 1} or {-1, 1}.

    y_prob : array, shape=(n_sample,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions
        (as returned by “decision_function” or "predict_proba" on some classifiers).

    sample_weight : array, shape = (n_samples,), optional
        Sample weights

    scores_metric_funcs : dict, optional
        Collection of metrics to evaluate once on the non-thresholed y_prob prediction.
        Dictionary which keys are metric names and value is a callable
        with the signature `scorer(y_true, y_pred, **kwargs)`.
        If not provided, uses the metric specified in `SCORES_METRIC_FUNCS`.

    Returns
    -------
    pd.Series: shape=(len(scores_metric_funcs),).
        Index names are scores_metric_funcs.keys().
        Holding metric evaluations of each metric (once) on the
        probabilities/scores (non-thresholed prediction) provided in y_prob.
    """
    scores_metric_funcs = scores_metric_funcs or SCORES_METRIC_FUNCS.copy()
    scores = {}
    for metric_name, metric_func in scores_metric_funcs.items():
        scores[metric_name] = metric_func(y_true, y_prob, sample_weight=sample_weight)
    scores = pd.Series(scores)
    return scores


def _plot_performance_curve(scores, x_axis_metric_name, y_axis_metric_name,
                            curve_type, pos_class_prevalence=None):
    """

    Parameters
    ----------
    scores : pd.DataFrame shape=(n_threshold, n_metrics).
        Column names are different metrics.
        Holding evaluations of each metric for each binarization of thresholds.

    x_axis_metric_name : str
        The name of the metric that would be plotted on the x-axis.
        For ROC it is probably "fpr", and for PR it is "recall".

    y_axis_metric_name : str
        The name of the metric that would be plotted on the y-axis.
        For ROC it is probably "tpr", and for PR it is "precision".

    curve_type : str
        Either "ROC" or "PR".

    pos_class_prevalence : float
        Prevalence of the positive class.
        This is needed for the chance-line of PR plot.
        If not provided (and curve_type="PR") chance line won't be plotted.

    Returns
    -------
    bokeh.plotting.figure:
            Bokeh figure object with ROC curve plotted on
            and interactive hover-tool showing evaluations at each point on curve.
    """

    hover = _create_hover_tool(scores.columns, curve_type,
                               x_axis_metric_name, y_axis_metric_name)

    # Plot:
    axis_range = (0 - PLOT_LIMIT_PAD, 1 + PLOT_LIMIT_PAD)
    p = figure(title="{} Curve".format(curve_type),
               plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT,
               x_range=axis_range, y_range=axis_range)

    p.add_tools(hover)

    # Plot main curve:
    auc = metrics.auc(scores[x_axis_metric_name], scores[y_axis_metric_name])
    p.line(x=x_axis_metric_name, y=y_axis_metric_name,
           legend="AUC: {:.3f}".format(auc),
           source=ColumnDataSource(scores),
           line_width=PLOT_LINE_WIDTH, name=curve_type)

    # Plot chance line:
    x_chance_line, y_chance_line = _get_chance_curve(curve_type, pos_class_prevalence)
    if x_chance_line and y_chance_line:
        p.line(x_chance_line, y_chance_line,
               line_dash="dashed", line_color="grey", line_width=PLOT_LINE_WIDTH,
               legend="Chance")

    # Edit:
    p.legend.location = "bottom_right" if curve_type == "ROC" else "bottom_left"
    p.xaxis.axis_label = x_axis_metric_name
    p.yaxis.axis_label = y_axis_metric_name
    return p


def _create_hover_tool(metrics_names, curve_type,
                       x_axis_metric_name, y_axis_metric_name):
    """
    Creates a Bokeh hover-tool that shows the evaluated metrics for any
     given point on the plot.

    Parameters
    ----------
    metrics_names : pd.Index
        Names of metric functions.
        These are the column names of the DataFrame returned by
         _calc_scores_per_thresh.

    curve_type : {'ROC', 'PR'}
        What type of curve is plotted.

    x_axis_metric_name : str
        The name of the metric that would be plotted on the x-axis.
        For ROC it is probably "fpr", and for PR it is "recall".

    y_axis_metric_name : str
        The name of the metric that would be plotted on the y-axis.
        For ROC it is probably "tpr", and for PR it is "precision".

    Returns
    -------
        bokeh.models.HoverTool
    """
    # Check whether confusion was calculated by checking if it has its 4 components
    conf_matrix_calculated = all([m in metrics_names for m in CONF_MATRIX_COMPONENTS])

    # Create hover tooltips:
    hover_tooltips = [("threshold", "@threshold"),
                      ("({x_metric},{y_metric})".format(x_metric=x_axis_metric_name,
                                                        y_metric=y_axis_metric_name),
                       "(@{x_metric}, @{y_metric})".format(x_metric=x_axis_metric_name,
                                                           y_metric=y_axis_metric_name))]

    # Add all scores to tooltip except the ones already in it:
    for score in metrics_names.difference(["threshold", x_axis_metric_name, y_axis_metric_name,  # Already added
                                           *CONF_MATRIX_COMPONENTS]):  # Confusion matrix will be added separately
        hover_tooltips.append((str(score), "@{}".format(score)))

    # Add confusion matrix to tooltip:
    if conf_matrix_calculated:
        hover_tooltips.append(("conf_matrix", CONF_MATRIX_FORMATTING))
    hover = HoverTool(tooltips=hover_tooltips,
                      names=[curve_type], mode="vline")

    return hover


def _get_chance_curve(curve_type, pos_class_prevalence=None):
    """
    Get the x values and y values of the curve representing chance
     prediction.

    For ROC curves, chance prediction is the diagonal x=y for which AUC
     is 0.5.
    For PR curves, chance prediction is the horizontal line at the
     positive-class prevalence in the data.

    Parameters
    ----------
    curve_type : str
        Either "ROC" or "PR". Other types are not supported (won't be plotted).

    pos_class_prevalence : float
        Prevalence of the positive class.
        This is needed for the chance-line of PR plot.
        If not provided (and curve_type="PR") chance line won't be plotted.

    Returns
    -------
    x_line : list
        x axis values of the chance line
    y_line : list
        y axis values of the chance line
    """
    x_line = [0, 1]
    if curve_type == "ROC":
        y_line = [0, 1]
    elif curve_type == "PR" and pos_class_prevalence is not None:
        y_line = [pos_class_prevalence, pos_class_prevalence]
    else:
        y_line = []
    return x_line, y_line

