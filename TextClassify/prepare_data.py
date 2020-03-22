# encoding: utf-8
import os
import jieba
from multiprocessing import Process, Queue
import multiprocessing


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


# datadir include category subdir
# return tuple list [(filepath, category)]
def get_data_tuples(datadir,categories,operation,cnt_per_type):
    tuples = []
    for _,dirs,_ in os.walk(datadir):
        for d in dirs:
            if d in categories:
                files = os.listdir(datadir+'/'+d)
                if operation == 'train':
                    for f in files[:cnt_per_type]:
                        f = datadir+'/'+d+'/'+f
                        tuples.append((f,d))
                if operation == 'test':
                    for f in files[-cnt_per_type-1:-1]:
                        f = datadir + '/' + d + '/'+f
                        tuples.append((f,d))

    return tuples


def get_stop_words(fpath):
    ignore_words= []
    with open(fpath,'r',encoding='utf-8') as fstop:
        for l in fstop.readlines():
            l = l.replace('\n','')
            ignore_words.append(l)

    return ignore_words


def cleanup_sentence(q,ftuples,stop_words):
    for f,k in ftuples:
        with open(f,'r',encoding='utf-8') as fr:
            text = fr.read().lower()
            text = text.replace("\t"," ").replace("\n"," ")
            seg_text = jieba.cut(text)
            segout = " ".join(seg_text)
            words = segout.split()
            outline = []  # " ".join(outline.split())
            for i in words:
                if i not in stop_words:
                    outline.append(i)
            outline = " ".join(outline) + "\t__label__" + k + "\n"
            q.put(outline)
            # print("segment file: %s" % f)
            # print(outline)
            # break


if __name__ == '__main__':
    stop_words = get_stop_words('./stopwords.dat')

    train_categories = ['财经', '股票', '科技', '社会', '游戏']
    test_categories = train_categories

    train_tuples = get_data_tuples('../data/THUCNews', train_categories, 'train', 1000)
    test_tuples = get_data_tuples('../data/THUCNews', test_categories, 'test', 200)

    q = Queue()
    procs = []
    proc_cnt = multiprocessing.cpu_count()

    if len(train_tuples) <= proc_cnt:
        proc_cnt = len(train_tuples)
    task_piece = len(train_tuples) / proc_cnt
    task_reserve = len(train_tuples) % proc_cnt

    e = 0
    for i in range(multiprocessing.cpu_count()):
        b = e
        e += task_piece
        if i < task_reserve:
            e += 1
        p = Process(target=cleanup_sentence, args=(q,train_tuples[int(b):int(e)],stop_words))
        p.start()
        procs.append(p)

    with open("news_train.txt", "w", encoding='utf-8') as ftrain:
        i = 0
        while i < len(train_tuples):
            line = q.get()
            ftrain.write(line)
            #ftrain.flush()
            i += 1

    for t in procs:
        t.join()
    procs.clear()

    if len(test_tuples) <= proc_cnt:
        proc_cnt = len(test_tuples)
    task_piece = len(test_tuples) / proc_cnt
    task_reserve = len(test_tuples) % proc_cnt

    e = 0
    for i in range(multiprocessing.cpu_count()):
        b = e
        e += task_piece
        if i < task_reserve:
            e += 1
        p = Process(target=cleanup_sentence, args=(q,test_tuples[int(b):int(e)],stop_words))
        p.start()
        procs.append(p)

    with open("news_test.txt", "w", encoding='utf-8') as f:
        i = 0
        while i < len(test_tuples):
            line = q.get()
            f.write(line)
            i += 1

    for t in procs:
        t.join()
    procs.clear()
