import numpy as np
from sklearn.preprocessing import LabelEncoder
from .logger import logger



def check_targets(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of samples")
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays")
    
    unique_values = np.union1d(y_true, y_pred)
    
    if len(unique_values) == 2:
        y_type = "binary"
    else:
        y_type = "multiclass"
    
    return y_type, y_true, y_pred


def multilabel_confusion_matrix(
    y_true,
    y_pred,
    sample_weight=None,
    labels=None,
    samplewise=False,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim != 1:
            raise ValueError(f"sample_weight should be 1D, but got shape {sample_weight.shape}.")

    present_labels = np.unique(np.concatenate([y_true.ravel(), y_pred.ravel()]))  # Labels present in data

    if labels is None:
        labels = present_labels
    else:
        labels = np.asarray(labels)
        labels = np.concatenate([labels, np.setdiff1d(present_labels, labels, assume_unique=True)])

    n_labels = len(labels)

    if y_true.ndim == 1:
        if samplewise:
            raise ValueError("Samplewise metrics are not available for single-label classification.")

        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)

        tp = y_true == y_pred
        tp_bins = y_true[tp]

        if tp_bins.shape[0]:
            tp_sum = np.bincount(tp_bins, weights=sample_weight[tp] if sample_weight is not None else None, minlength=n_labels)
        else:
            tp_sum = np.zeros(n_labels, dtype=int)

        if y_pred.shape[0]:
            pred_sum = np.bincount(y_pred, weights=sample_weight, minlength=n_labels)
        else:
            pred_sum = np.zeros(n_labels, dtype=int)

        if y_true.shape[0]:
            true_sum = np.bincount(y_true, weights=sample_weight, minlength=n_labels)
        else:
            true_sum = np.zeros(n_labels, dtype=int)

    else:
        sum_axis = 1 if samplewise else 0

        if labels.shape != present_labels.shape or np.any(labels != present_labels):
            if np.max(labels) > np.max(present_labels) or np.min(labels) < 0:
                raise ValueError("All labels must be within [0, n_labels).")

        if n_labels:
            y_true = y_true[:, :n_labels]
            y_pred = y_pred[:, :n_labels]

        true_and_pred = y_true * y_pred
        tp_sum = np.sum(true_and_pred, axis=sum_axis)
        pred_sum = np.sum(y_pred, axis=sum_axis)
        true_sum = np.sum(y_true, axis=sum_axis)

    tp_sum = np.asarray(tp_sum)
    pred_sum = np.asarray(pred_sum)
    true_sum = np.asarray(true_sum)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tn = y_true.shape[0] - tp_sum - fp - fn if not samplewise else y_true.shape[1] - tp_sum - fp - fn

    return np.stack([tn, fp, fn, tp_sum], axis=-1).reshape(-1, 2, 2)


def precision_recall_fscore_support(
    y_true,
    y_pred,
    beta=1.0,
    labels=None,
    average=None,
    sample_weight=None,
):
    MCM = multilabel_confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels)
    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]
    
    if average == "micro":
        tp_sum = np.array([np.sum(tp_sum)])
        pred_sum = np.array([np.sum(pred_sum)])
        true_sum = np.array([np.sum(true_sum)])
    
    precision = np.divide(tp_sum, pred_sum, where=pred_sum != 0)
    recall = np.divide(tp_sum, true_sum, where=true_sum != 0)
    
    if np.isposinf(beta):
        f_score = recall
    elif beta == 0:
        f_score = precision
    else:
        beta2 = beta**2
        denom = beta2 * true_sum + pred_sum
        f_score = np.divide((1 + beta2) * tp_sum, denom, where=denom != 0)
    
    if average == "weighted":
        weights = true_sum
    elif average == "samples":
        weights = sample_weight
    else:
        weights = None
    
    if average is not None:
        precision = np.average(precision, weights=weights) if weights is not None else np.mean(precision)
        recall = np.average(recall, weights=weights) if weights is not None else np.mean(recall)
        f_score = np.average(f_score, weights=weights) if weights is not None else np.mean(f_score)
        true_sum = None
    
    return precision, recall, f_score, true_sum


def classification_report(
    y_true,
    y_pred,
    labels=None,
    target_names=None,
    sample_weight=None,
    print_report=True,
):
    y_type, y_true, y_pred = check_targets(y_true, y_pred)

    if labels is None:
        labels = np.unique([y_true, y_pred])
        labels_given = False
    else:
        labels = np.asarray(labels)
        labels_given = True

    micro_is_accuracy = not labels_given or (set(labels) >= set(np.unique([y_true, y_pred])))

    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            logger.warning(f"labels size, {len(labels)}, does not match size of target_names, {len(target_names)}")
        else:
            raise ValueError(
                "Number of classes, {0}, does not match size of "
                "target_names, {1}. Try specifying the labels "
                "parameter".format(len(labels), len(target_names))
            )
    if target_names is None:
        target_names = ["%s" % l for l in labels]

    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
    )

    headers = ["precision", "recall", "f1-score", "samples"]
    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    report_list = [{}, {}]
    for label, precision, recall, fscore, support in rows:
        report_list[0][label] = {
            "precision": float(f"{precision:.4f}"),
            "recall": float(f"{recall:.4f}"),
            "f1-score": float(f"{fscore:.4f}"),
            "samples": float(f"{support:.1f}")
        }

    for average in average_options:
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            sample_weight=sample_weight,
        )
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]
        report_list[1][line_heading] = {
            "precision": float(f"{avg[0]:.4f}"),
            "recall": float(f"{avg[1]:.4f}"),
            "f1-score": float(f"{avg[2]:.4f}"),
            "samples": float(f"{avg[3]:.1f}")
        }

    if "accuracy" in report_list[1]:
        report_list[1]["accuracy"] = report_list[1]["accuracy"]["precision"]

    if print_report:
        print(format_classification_report(report_list))
    return report_list



def format_classification_report(report_list):
    headers = ["precision", "recall", "f1-score", "samples"]
    longest_label = max(len(label) for label in report_list[0])
    width = max(longest_label, len("weighted avg"))

    # Header
    report = f"{'':>{width}} " + " ".join(f"{h:>10}" for h in headers) + "\n\n"

    # Rows for each class
    for label, scores in report_list[0].items():
        row = f"{label:>{width}} " + " ".join(
            f"{scores[h]:>10.4f}" if h != "samples" else f"{int(scores[h]):>10}" 
            for h in headers
        )
        report += row + "\n"

    # Accuracy row
    report += f"\n{'accuracy':>{width}} {'':>10} {'':>10} {report_list[1]['accuracy']:>10.4f} {int(report_list[1]['macro avg']['samples']):>10}\n"
    # Rows for macro avg, weighted avg
    for avg_label in ["macro avg", "weighted avg"]:
        if avg_label in report_list[1]:
            scores = report_list[1][avg_label]
            row = f"{avg_label:>{width}} " + " ".join(
                f"{scores[h]:>10.4f}" if h != "samples" else f"{int(scores[h]):>10}" 
                for h in headers
            )
            report += row + "\n"

    return report
