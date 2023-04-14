# coding: utf-8
from mindspore import nn

from src.train_and_evaluate import *
from src.models import *
from src.lr_schedule import dynamic_lr
from src.expressions_transfer import *
import time

from src.expressions_transfer import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


batch_size = 150
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
n_layers = 2

data_all = load_raw_data("/home/ma-user/modelarts/user-job-dir/seq2tree/data/Math_23K_interpretable.json") # 6666 instances
data_train = load_raw_data("/home/ma-user/modelarts/user-job-dir/seq2tree/data/Math_23K_train.json") # 5332 instances
data_test = load_raw_data("/home/ma-user/modelarts/user-job-dir/seq2tree/data/Math_23K_test.json") # 1334 instances

_, generate_nums, copy_nums = transfer_num(data_all)
pairs_train, _, _ = transfer_num(data_train)
pairs_test, _, _ = transfer_num(data_test)


temp_pairs = []
for p in pairs_train:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_train = temp_pairs

temp_pairs = []
for p in pairs_test:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs_test = temp_pairs

#
# fold_size = int(len(pairs) * 0.2)
# fold_pairs = []
# for split_fold in range(4):
#     fold_start = fold_size * split_fold
#     fold_end = fold_size * (split_fold + 1)
#     fold_pairs.append(pairs[fold_start:fold_end])
# fold_pairs.append(pairs[(fold_size * 4):])
#
best_acc_fold = []
#
# for fold in range(5):
#     pairs_tested = []
#     pairs_trained = []
#     for fold_t in range(5):
#         if fold_t == fold:
#             pairs_tested += fold_pairs[fold_t]
#         else:
#             pairs_trained += fold_pairs[fold_t]
pairs_trained = pairs_train
pairs_tested = pairs_test
input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)
# Initialize models
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder.set_train()
predict.set_train()
generate.set_train()
merge.set_train()


withloss = Seq2TreeWithLossCell(encoder, predict, generate, merge, output_lang.word2index["UNK"], output_lang.num_start)

warmup_step = 300
warmup_ratio = 0.33
base_step = len(data_all)//batch_size + 1
lr_dynamic = dynamic_lr(learning_rate, n_epochs, warmup_step, warmup_ratio, base_step)

optimizer = nn.Adam(withloss.trainable_params(), learning_rate=lr_dynamic, weight_decay=weight_decay)

# trainonestep = S2TTrainOneStepCell(withloss, optimizer)
trainonestep = nn.TrainOneStepCell(withloss, optimizer)


generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in range(len(input_lengths)):

        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, trainonestep, num_pos_batches[idx])
        loss_total += loss
        print(loss)
        if(idx%20==0):
            print('batch %d time: %f'%(idx, time.time()-start))

    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    # if epoch % 10 == 0 or epoch > n_epochs - 5:
    #     value_ac = 0
    #     equation_ac = 0
    #     eval_total = 0
    #     start = time.time()
    #     for test_batch in test_pairs:
    #         test_res = evaluate_tree(trainonestep.network.encoder, trainonestep.network.predict, trainonestep.network.generate, trainonestep.network.merge, test_batch[0], test_batch[1], generate_num_ids, test_batch[5], output_lang.num_start)
    #         val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    #         if val_ac:
    #             value_ac += 1
    #         if equ_ac:
    #             equation_ac += 1
    #         eval_total += 1
    #     print(equation_ac, value_ac, eval_total)
    #     print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
    #     print("testing time", time_since(time.time() - start))
    #     print("------------------------------------------------------")
        
    #     if epoch == n_epochs - 1:
    #         best_acc_fold.append((equation_ac, value_ac, eval_total))

import mindspore as ms
ms.save_checkpoint(trainonestep, "./MyNet.ckpt")
# a, b, c = 0, 0, 0
# for bl in range(len(best_acc_fold)):
#     a += best_acc_fold[bl][0]
#     b += best_acc_fold[bl][1]
#     c += best_acc_fold[bl][2]
#     print(best_acc_fold[bl])
# print(a / float(c), b / float(c))
