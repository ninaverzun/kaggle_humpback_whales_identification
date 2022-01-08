import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)

LABELS = 'ensembling/train.csv'
SAMPLE_SUB = 'ensembling/sample_submission.csv'


def get_train_df():
    train = pd.read_csv(LABELS)
    criteria = train['Id'] != 'new_whale'
    whales_train = train[criteria]
    return whales_train


def remove_new_whale(unique_labels):
    labels_dict = dict()
    labels_list = []

    for i in range(len(unique_labels)):
        labels_dict[unique_labels[i]] = i
        labels_list.append(unique_labels[i])

    print("Number of classes: {}".format(len(unique_labels)))

    print(np.shape(labels_list))
    labels_list = np.array(labels_list)
    return labels_list, labels_dict


def add_new_whale_to_predictions(preds):
    sorted_preds = np.sort(preds)
    avg_of_max_predictions = np.average(sorted_preds[:, -1:])
    print("Average of max probabilities column:" + str(avg_of_max_predictions))
    best_threshold = avg_of_max_predictions
    # print(np.shape(preds))
    shape_to_add = (np.shape(preds)[0], 1)
    # Add a column with the best threshold probability to the predictions
    column_to_add = np.zeros(shape_to_add) + best_threshold
    predictions_w_new_whale = np.concatenate([column_to_add, preds], axis=1)
    return predictions_w_new_whale


def create_results_csv(preds, labels_with_new_whale, test_file_names, output_filename):
    sample_df = pd.read_csv(SAMPLE_SUB)
    sample_images = list(sample_df.Image)

    # print(test_file_names[:7])
    pred_list = [[labels_with_new_whale[i] for i in p.argsort()[-5:][::-1]] for p in preds]

    pred_dic = dict((key, value) for (key, value) in zip(test_file_names, pred_list))
    pred_list_for_test = [' '.join(pred_dic[id]) for id in sample_images]

    # print(np.shape(pred_list))
    # print(np.shape(test_file_names))
    df = pd.DataFrame({'Image': sample_images, 'Id': pred_list_for_test})
    df.to_csv(output_filename, header=True, index=False)
    return df


def average_predictions(preds_list):
    predictions = np.stack(preds_list, axis=-1)
    # print("shape after stacking: " + str(np.shape(predictions)))
    predictions = predictions.mean(axis=-1)
    # print("shape after mean: " + str(np.shape(predictions)))
    # print(np.shape(predictions))
    p = add_new_whale_to_predictions(predictions)
    return p


def create_submission_csv(labels, file_name, preds):
    test_files = np.load("ensembling/filenames.npy")
    test_df = create_results_csv(preds, labels, test_files, file_name)
    print(test_df[:10])


train_df = get_train_df()
unique_labels = np.unique(train_df.Id.values)
labels_list, labels_dict = remove_new_whale(unique_labels)
labels_with_new_whale = np.concatenate((['new_whale'], labels_list), axis=0)

p0_426 = np.load('ensembling\predictions_resnet50_lb_0_426.npy')
p0_515 = np.load('ensembling\predictions_inceptionresnetv2_lb_0_515.npy')
p0_449 = np.load('ensembling\predictions _xception_lb_0_449.npy')
p0_423 = np.load('ensembling\predictions_xception_adagrad_lb_0_423.npy')

predictions = [p0_426, p0_515]
predictions2 = [p0_426, p0_515, p0_449]
predictions3 = [p0_426, p0_515, p0_449, p0_423]
predictions4 = [p0_515, p0_449]

print("========================================== Ensembling Resnet50 (size 100*100) and InceptionResnetV2 (size 128*128) predictions ")
p1 = average_predictions(predictions)
create_submission_csv(labels_with_new_whale, "ensembling\submission_ensembling.csv", p1)

print("========================================== Ensembling: \n1) Resnet50(size 100*100) \n2) InceptionResnetV2 (size 128*128) \n3) Xception (size 128*128) "
      "\npredictions")
p2 = average_predictions(predictions2)
create_submission_csv(labels_with_new_whale, "ensembling\submission_ensembling_2.csv", p2)

print("=========================================== Ensembling: \n1) Resnet50(size 100*100) \n2) InceptionResnetV2 (size 128*128) \n3) Xception (size 128*128) "
      "\n4) Xception Adagrad optimizer(size 128*128) \npredictions")
p3 = average_predictions(predictions3)
create_submission_csv(labels_with_new_whale, "ensembling\submission_ensembling_3.csv", p3)

print("========================================== Ensembling top 2 scores: \n1) InceptionResnetV2 (size 128*128) \n2) Xception (size 128*128) \npredictions")
p4 = average_predictions(predictions4)
create_submission_csv(labels_with_new_whale, "ensembling\submission_ensembling_4.csv", p4)
