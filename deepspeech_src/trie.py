from typing import Tuple
import _pickle as pickle
import re


class TrieNode(object):
    """
    Our trie node implementation. Very basic. but does the job
    """

    def __init__(self, char: str):
        self.char = char
        self.children = {}
        # Is it the last character of the word.`
        self.word_finished = False
        # How many times this character appeared in the addition process
        self.counter = 1


def add(root, word: str):
    """
    Adding a word in the trie structure
    """
    node = root
    node.counter+=1
    for char in word:
        # print(char)
        found_in_child = False
        # Search for the character in the children of the present `node`
        if char in node.children:
            node.children[char].counter += 1
            # print(char, node.children[char].counter)
            found_in_child = True
            node = node.children[char]
        # We did not find it so add a new chlid
        if not found_in_child:
            new_node = TrieNode(char)
            node.children[char] = new_node
            # And then point node to the new child
            node = new_node
    # Everything finished. Mark it as the end of a word.
    node.word_finished = True


def probability(root, prefix):
    node = root

    if not root.children:
        return 0
    for char in prefix:
        char_not_found = True
        if char in node.children:
            # We found the char existing in the child.
            char_not_found = False
            # Assign node as the child containing the char and break
            parent_node = node
            node = node.children[char]

        if char_not_found:
            return 0
    # Well, we are here means we have found the prefix. Return true to indicate that
    # And also the counter of the last node. This indicates how many words have this
    # prefix
    # print(parent_node.counter)
    # print(node.counter)
    denominator = parent_node.counter - int(parent_node.word_finished)
    return node.counter/denominator


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    root = TrieNode('*')
    print(root.__class__.__module__)
           
    if root.__class__.__module__ == "__main__":
        print("Here")
        
        from trie import TrieNode  
        path = "/home/rjd2551/Speech/Gujarati/"
        root = TrieNode('*')
        print(root.__class__.__module__)
        k = load_obj(path + "gu-lexicon")
        for word in k:
            add(root, k[word])
        #transcript_path = "D:\Thesis\Microsoft\microsoftspeechcorpusindianlanguages\gu-in-Train\\transcription.txt"
        transcript_path = path+"transcription.txt"
        with open(transcript_path, "r", encoding="utf-8") as fi:
            for line in fi:
                content = line.split("\t")
                sentence = content[1]
                for word in re.findall('\S+', sentence):
                    # print(word)
                    add(root, k[word])
        with open(path +'gu-trie.pkl', 'wb') as output:
            pickle.dump(root, output, -1)
  
    #pass
    # root = TrieNode('*')
    # k = load_obj("gu-lexicon")
    # for word in k:
    #     add(root, k[word])
    # transcript_path = "D:\Thesis\Microsoft\microsoftspeechcorpusindianlanguages\gu-in-Train\\transcription.txt"
    #
    # with open(transcript_path, "r", encoding="utf-8") as fi:
    #     for line in fi:
    #         content = line.split("\t")
    #         sentence = content[1]
    #         for word in re.findall('\S+', sentence):
    #             # print(word)
    #             add(root, k[word])
    # add(root, "hackathon")
    # add(root, 'hack')
    # add(root, "archer")
    # with open('company_data.pkl', 'wb') as output:
    #     pickle.dump(root, output, -1)
    # with open('company_data.pkl', 'rb') as input:
    #     root = pickle.load(input)
    # print(root.children)
    # print(probability(root, 'hackat'))
    #TrieNode.__module__ = "trie"  
    #with open(path +'gu-trie.pkl', 'wb') as output:
    #    pickle.dump(root, output, -1)
    # with open('gu_trie.pkl', 'rb') as input:
    #     root = pickle.load(input)
    # print(probability(root, ["n", "a", "h", "ii", "q"]))