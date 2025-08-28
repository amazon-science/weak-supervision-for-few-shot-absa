# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement


"""
Extract what it was predicted inside seq. Use :param task to know how to match against it
We strip aspect terms and opinion terms (e.g. from rest16/train.jsonl - `nearing $ 7 ` vs `nearing $ 7` -- notice the trailing space)
Spaces in the beginning and at the end do not add any value, so we just strip them (also, it is not consistent)
"""


def extract_spans_para(task, seq, lower=False):
    task_preds = []
    sents = [s.strip() for s in seq.split("[SSEP]")]
    if task == "task1":
        for s in sents:
            try:
                at = s
                # if the aspect term is implicit
                # if at.lower() == "it":
                # at = "NULL"
            except ValueError:
                at = ""
            at = at.strip()
            if lower:
                task_preds.append(at.lower())
            else:
                task_preds.append(at)
    elif task == "task2":
        for s in sents:
            try:
                "$AT is $S"
                at, sp = s.split(" is ")
                # if the aspect term is implicit
                # if at.lower() == "it":
                # at = "NULL"
            except ValueError:
                at, sp = "", ""
            at = at.strip()
            if lower:
                task_preds.append((at.lower(), sp))
            else:
                task_preds.append((at, sp))
    elif task == "task3":
        for s in sents:
            try:
                "$AT is $S means $AC $S"
                "$AC $S because $AT is $S"
                if " means " in s:
                    at_sp, ac_sp = s.split(" means ")
                elif " because " in s:
                    ac_sp, at_sp = s.split(" because ")
                else:
                    raise ValueError("Invalid format!")
                at, sp = at_sp.split(" is ")
                # if at.lower() == "it":
                # at = "NULL"
                ac, sp = ac_sp.rsplit(" ", 1)
            except ValueError:
                at, ac, sp = "", "", ""
            at = at.strip()
            if lower:
                task_preds.append((at.lower(), ac, sp))
            else:
                task_preds.append((at, ac, sp))
    elif task == "task4":
        for s in sents:
            try:
                "$AT is $OT means $DOMAIN is $S"
                "$DOMAIN is $S because $AT is $OT"
                if " means " in s:
                    at_ot, ac_sp = s.split(" means ")
                elif " because " in s:
                    ac_sp, at_ot = s.split(" because ")
                else:
                    raise ValueError("Invalid format!")
                at, ot = at_ot.split(" is ")
                _, sp = ac_sp.rsplit(" ", 1)
                # if the aspect term is implicit
                # if at.lower() == "it":
                # at = "NULL"
            except ValueError:
                at, sp, ot = "", "", ""
            at = at.strip()
            ot = ot.strip()
            if lower:
                task_preds.append((at.lower(), sp, ot.lower()))
            else:
                task_preds.append((at, sp, ot))
    elif task == "task5":
        for s in sents:
            try:
                "$AT is $OT means $AC $S"
                "$AC $S because $AT is $OT"
                if " means " in s:
                    at_ot, ac_sp = s.split(" means ")
                elif " because " in s:
                    ac_sp, at_ot = s.split(" because ")
                else:
                    raise ValueError("Invalid format!")
                if " is " in at_ot:
                    at, ot = at_ot.split(" is ")
                elif "is " in at_ot:
                    at, ot = at_ot.split("is ")
                elif " is" in at_ot:
                    at, ot = at_ot.split(" is")
                else:
                    raise ValueError("Invalid format")
                ac, sp = ac_sp.rsplit(" ", 1)
                # if the aspect term is implicit
                # if at.lower() == "it":
                # at = "NULL"
            except ValueError:
                ac, at, sp, ot = "", "", "", ""
            at = at.strip()
            ot = ot.strip()
            if lower:
                task_preds.append((ac, at.lower(), sp, ot.lower()))
            else:
                task_preds.append((ac, at, sp, ot))
    elif task == "ate":
        for s in sents:
            try:
                at = s
            except ValueError:
                at = ""
            task_preds.append(at)
    elif task == "ote":
        for s in sents:
            try:
                ot = s
            except ValueError:
                ot = ""
            task_preds.append(ot)
    elif task == "ate_ote":
        for s in sents:
            try:
                "$AT is described as $OT"
                if "is described as" in s:
                    at, ot = [x.strip() for x in s.split("is described as")]
                else:
                    raise ValueError("Invalid format!")
            except ValueError:
                at, ot = "", ""
            task_preds.append((at.strip(), ot.strip()))
    elif task == "ate_sent":
        for s in sents:
            try:
                "$AT is $S"
                if " is " in s:
                    at, sp = s.split(" is ")
                elif "is " in s:  # To allow handling `matride d'`
                    at, sp = s.split("is ")
                elif " is" in s:
                    at, sp = s.split(" is")
                else:
                    raise ValueError("Invalid format!")
            except ValueError:
                at, sp = "", ""
            task_preds.append((at.strip(), sp.strip()))
    elif task == "ate_ote_sent":
        for s in sents:
            try:
                "$AT is $S because it is described as $OT"
                if "is described as" in s:
                    at_sp, ot = [
                        x.strip() for x in s.split("because it is described as")
                    ]
                    at, sp = at_sp.split(" is ")
                else:
                    raise ValueError("Invalid format!")
            except ValueError:
                at, sp, ot = "", "", ""
            task_preds.append((at.strip(), sp.strip(), ot.strip()))
    else:
        raise NotImplementedError(f"Not implemented for task {task}")

    return task_preds


"""
:param pred_pt -> 
:param gold_pt -> 
"""


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])
        gold_pt_copy = gold_pt[i].copy()
        for t in list(pred_pt[i]):
            if t in gold_pt_copy:
                gold_pt_copy.remove(t)
                n_tp += 1

    # print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision != 0 or recall != 0
        else 0
    )
    scores = {"precision": precision, "recall": recall, "f1": f1}
    if f1 > 1:
        print(pred_pt, gold_pt, scores)
    return scores


def get_task_name_from_input(input, templates):
    # given the input, we need to return the name of the task
    # we use the fact that dictionaries are ordered (as of python 3.7)
    for template_id in reversed(list(templates.keys())):
        phrases = templates[template_id]["phrases"]
        for phrase in phrases:
            if phrase in input:
                return template_id


def compute_scores(
    input_seqs,
    pred_seqs,
    gold_seqs,
    templates,
    verbose=False,
    lower=False,
    default_task=None,
):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    assert len(pred_seqs) == len(input_seqs)
    num_samples = len(gold_seqs)

    task_names = sorted(list(templates.keys()))
    all_scores = {}
    all_inputs = {}
    all_preds = {}
    all_labels = {}
    for task_name in task_names:
        all_inputs[task_name] = []
        all_preds[task_name] = []
        all_labels[task_name] = []

    none_task_counts = 0
    for i in range(num_samples):
        # aspect, sentiment, opinion and category
        task_name = default_task or get_task_name_from_input(input_seqs[i], templates)
        if not task_name:
            print(input_seqs[i])
            # exit()
            none_task_counts += 1
            continue
        gold_list = extract_spans_para(task_name, gold_seqs[i], lower)
        pred_list = extract_spans_para(task_name, pred_seqs[i], lower)

        all_inputs[task_name].append(input_seqs[i])
        all_labels[task_name].append(gold_list)
        all_preds[task_name].append(pred_list)

    for task_name in task_names:
        all_scores[task_name] = compute_f1_scores(
            all_preds[task_name], all_labels[task_name]
        )
        # all_scores[task_name] = compute_f1_scores([[y.lower() for y in x] for x in all_preds[task_name]], [[y.lower() for y in x] for x in all_labels[task_name]])
        if verbose:
            print(f"\nResults for {task_name}:")
            print(all_scores[task_name])

    if verbose:
        print("none_task_counts: ", none_task_counts)

    return all_scores, all_inputs, all_labels, all_preds
