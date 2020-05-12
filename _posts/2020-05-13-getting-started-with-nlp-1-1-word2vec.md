---
layout: post
title:  "Getting Started with NLP : 1.1 | Word 2Vec |"
date:   2020-05-13 03:23:13 +0530
categories: 
---
# Getting Started with NLP : 1.1 | Word2Vec |

## Word2Vec

 - Word2Vec is a **word-representing technique** which makes use of **distributed representation** of words and phrases.

![Word Embeddings](https://developers.google.com/machine-learning/crash-course/images/linear-relationships.svg)

Explaining the same in layman terms : 

 - Word2Vec is a technique which converts/represents a word in a vocabulary into a float vector (of constant size) which can represent the sematic/syntactic meaning of the particular word. 

Basically it makes sense of the human-language to our machine so that we make build our Machine Learning Models.  

### History

This was the first-of-its-kind word embedding model which used neural networks to build the embedding model. Published by [Google Reasearch Team](https://arxiv.org/pdf/1301.3781.pdf). This paper consisted of techniques **Skip-Gram Model** and **CBOW Model**.


### Distributed Representations

During the time when this was published, most of the word-representing methods were of **local nature**.

#### Understanding Local Nature : 

Let us assume a neural net.

- This meant that each neuron is representing one **entity** i.e  for example neuron had a **one-to-one relationship with any entity**.

For example, we want to represent Volkswagen logo and to make matter much simpler let's assume the logo is as below.
The neural network is attached side-by-side.

![]()
<div align="center">
<img src="https://i.imgur.com/93goEbg.png" >

</div>


In the neural network we can see that every neuron in represents **exclusively one entity**, here entity can be a circle, triangle, letter-V, blue.

1. If we wanted to represent the logo we would have to activate all the 4 neurons at once. Seems simple and easy, right?
2. But what if we need to represent something like below? Same as above, right?

<div align="center">
<img src="https://i.imgur.com/ZuovTvz.png" >
</div>

I guess I could explain the serial nature of inputs and why local nature can't be a genaralized model (most of the times).
	
	Question : So what could be the solution?
	Answer : Can we make the neuron as Many-To-One relation? 

That is, a technique where a neuron would not just be related to one-entity and be **a part of a distributed network where every neuron effects the impact (weight) of an entity**; **distributed representations**. 

#### Understanding the Distributed Representation : 

In this representation the neuron has a many-to-one relation with entities i.e it could represent one-or-more entities. 

These form of representations are : 
1. Constructive Character
2. Ability to generalize automatically
3. Adaptability/Tunability to changing environments


## Word2Vec Algorithm

The word2vec is a technique where we can get our word vectors using 2 major techniques : 

1. Skip-Gram
2. Continuous Bag of Words (CBOW)

The most novel thing these algorithms had was that these had **no non-linear layer in the neural net architecture**, for which the neural net's were attractive for. But why?

The answer lies in the fact that the non-linear layer is the layer where the most computational complexity lies and for natural language modelling that was a bottleneck as we have to train the models on greater vocabulary for better models. This paper tried to explore simpler models and guess what, they hit the SOTA (State-of-the-Art).

### CBOW

The main principle behind this was **given a context the model had to predict target word**.
	
	"Where are you going Robert?"
	  [0]  [1] [2]  [3]   [4]
	
Consider a **context window of size 2** which would look like : 

| Word   | Left-Context  | Right-Context |
|---|---| --- |
| you  | [where, are]  | [going, robert]  |
| going  | [are, you]  | [robert]  |
| robert  | [you, going]  | [] |


So for a given context we predict a target word (probability) by training the shallow Neural Network. 

<div align="center">
<img src="https://i.imgur.com/Z4gNR7W.png" >
<p><b> CBOW Architecture </b></p>
</div>

This architecture has three layers : 

1. Input Layer : Has the context words as input.
2. Projection Layer : This is a replacement to the hidden-layer with no non-linear computation. This uses averaging of vectors and this projection layers is shared by every word i.e every word gets projected into the same position.

	- This is called a Bag-of-Words model as the order of words in the history does not influence the projection. We also use future words.

3. Output Layer : The word representation in a form of a vector.


### Skip-Gram 

 The main principle here is almost **reverse** of CBOW i.e **given a target word we predict the context**.

Basically, instead of predicting the current word based on the context, it tries to maximize the classification of a word based on another word in the same sentence.

Layman Terms : We are using one word in a sentence to correctly predict/classify another word in the same sentence.

**Technically** : 
	
- Use each current word as an input to a log-linear classifier with continuous projection layer, and predict words within a certain range (context size C) before and after the current word. 

<div align="center">
<img src="https://i.imgur.com/gruSd5v.png" >
<p><b> SkipGram Architecture </b></p>
</div>

It was seen that increasing the context range improves quality of the resulting word vectors, but it also increases the computational complexity. Since the more distant words are usually less related to the current
word than those close to it, the algorithm gives less weight to the distant words by sampling less from those words in the training examples.


# Implementation

We will try to utilize the pre-trained vectors on GoogleNews dataset. The pre-trained version mostly is better as it is trained on bigger vocabulary and for longer durations outputting a better and reliable model.

We can also train our own vectors, which I would leave that upto you by replicating this [beautiful kernel.](https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial#Training-the-model)


 - To load the pre-trained embedding : 
	 
	 - Load the pre-trained vector.
		 
			 import gensim.models.keyedvectors as word2vec
			 word2vec_dict = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

	- Initialize a embedding size (size of word-vector you need) and the embedding index (this will have the weights for every word in the vocabulary).
		
			embed_size = 300
			embedding_index = dict()
			# Store every word-weight in the vocabulary of loaded vector in our embedding_index
			for word in tqdm_notebook(word2vec_dict.wv.vocab):
		        embedding_index[word] = word2vec_dict.word_vec(word)
			 
			 print("Loaded {} word vectors.".format(len(embedding_index)))
			# Garbage Collector can help you clearing up unnecessary space
			gc.collect()
	
	- Now all of the words in your data might not be present in the pre-trained vector. For that we use random weights. So for replicating purpose i.e we replicate the same weights multiple times we run our code, we use the statistics mean, standard deviation of the embeddings to generate the randoms.
		 
			all_embs = np.stack(list(embedding_index.values()))
			emb_mean, emb_std = all_embs.mean(), all_embs.std()
			# Number of words in pre-trained vector
			nb_words = len(tokenizer.word_index) 
	
	- Generating our random-embedding matrix : 

			# We are going to set the embedding size to the pretrained dimension as we are replicating it.
			# The size would be Number of Words X Embedding Size
			embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
			gc.collect()
			# Now we have generated a random matrix lets fill it out with our own dictionary and the one with pretrained embeddings.

	- Now we have to fill the genereated embedding matrix for the words which are common in our data vs pre-trained data;  so we have to fill the   the values from the pretrained vector stored in embedding_index matrix.
				
			embedded_count = 0
			for word, i in tqdm_notebook(tokenizer.word_index.items()):
				i -= 1
				# Then we see whether the word is in Word2Vec dictionary, if yes get the pretrained weights.
				embedding_vector = embedding_index.get(word)
				# And store that in our embedding_matrix that we will use to train ML model.
				if embedding_vector is not None:
					embedding_matrix[i] = embedding_vector
					embedded_count += 1
			print("Total Embedded : {} common wrds".format(embedded_count))
			del embedding_index
			gc.collect()

	- Now we can use the embedding_matrix to build ML Models on-top of these.

We can put all of this in a function : 

<script src="https://gist.github.com/kranthik13/ee173fdc6d8b97fe313d47d4e8720d9f.js"></script>


I've published a [kernel which build an basic LSTM network on this embedding_matrix to give you a basic understanding of how to implement and use the weights.](https://www.kaggle.com/whoiskk/getting-started-with-nlp-1-1-word2vec/)


## Results of the Algorithm (Paper)

The paper proposed techniques for measuring the quality of the resulting vector representations.
These evaluation techniques were very intuitive yet unique and accurate.

They came up with a technique based on vector operations on words : 

1. Word 1 : King
2. Word 2 : Man
3. Word 3 : Woman

Vector Operation : 
	
	X = vector(”King”) - vector(”Man”) + vector(”Woman”)

And do you know what was the word-vector most closest to X?
		
	X = vector("Queen")
 
Seems interesting, right?

They made such rules and evaluated the word-representations on accuracies of ability to capture such syntatic and semantic relationship.

![Evaluation Rules](https://i.imgur.com/ZgeCoPu.png) 

## References : 

1. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

## Conclusion

I tried to explain the Word2Vec algorithms and work published in the paper and tried to utilize the pre-trained word vectors to train an LSTM (in the [kernel](https://www.kaggle.com/whoiskk/getting-started-with-nlp-1-1-word2vec/)) and evaluated the performance. 

