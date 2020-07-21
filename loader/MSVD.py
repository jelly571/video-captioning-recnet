import pandas as pd

from loader.data_loader import CustomVocab, CustomDataset, Corpus


class MSVDVocab(CustomVocab):
    """ MSVD Vocaburary """

    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[pd.notnull(df['Description'])]
        captions = df['Description'].values
        return captions

    def build(self):
        captions = self.load_captions()
        for caption in captions:
            #str.split()通过指定分隔符对字符串进行切片
            words = self.transform(caption)
            #计算最大的句子长度
            self.max_sentence_len = max(self.max_sentence_len, len(words))
            for word in words:
                #计算单词出现次数
                self.word_freq_dict[word] += 1
        #不重复的单词数(词汇)
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        #单词总数（重复）
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))
        #提取出不重复的单词
        keep_words = [ word for word, freq in self.word_freq_dict.items() if freq >= self.min_count ]
        
        for idx,word in enumerate(keep_words, len(self.word2idx)):   
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        #加了4个符号的不重复的单词数(词汇)
        self.n_vocabs = len(self.word2idx)
        self.n_words = sum([ self.word_freq_dict[word] for word in keep_words ])
        

class MSVDDataset(CustomDataset):
    """ MSVD Dataset """

    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[[ 'VideoID', 'Start', 'End', 'Description' ]]
        df = df[pd.notnull(df['Description'])]

        for video_id, start, end, caption in df.values:
            #用_连接起来id_start_end
            vid = "{}_{}_{}".format(video_id, start, end)
            self.captions[vid].append(caption)
        #print(self.captions)一个vid对应很多条caption


class MSVD(Corpus):
    """ MSVD Corpus """

    def __init__(self, C):
        super(MSVD, self).__init__(C, MSVDVocab, MSVDDataset)

