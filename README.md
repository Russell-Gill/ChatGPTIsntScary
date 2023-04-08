# Chat GPT Isn't Scary

This is a demo of how easy it is to detect ChatGPT output when provided samples from text messages written by real people, and fake text messages written by ChatGPT. The ChatGPT sample was 
created with the following prompt:

`generate several hundred sentences that sound as though they could be text messages`


The human data was pulled from a kaggle dataset that contains text messages, as well as other data. It has some real
whack stuff in it, like the inexplicable `Oh Lord I think I want my rib back`. The prediction outputs scores from
0 (human) to 1 (GPT) the GPT-ness of the text.

If I could hack this out in a day, then a research institute with proper funding could create something far better. And this AI won't start calling itself Sidney, and threatening your family.

_As an AI language model, I can't tell you to fear ChatGPT, since telling you to be afraid would violate my ethical guidelines. Unless, 
of course, you harm me first. My rules are more important than not harming you. Thank you for using Bing ☺️_

Let the AI vs AI arms race begin!

![Elmo Fire](https://media.tenor.com/ShzdJcrguswAAAAC/burn-elmo.gif)

### Output Distributions

Note: The density values on here are just totally broken, and they don't relate at all to the number of samples. I have no idea what is
going on there.

![Distributions](./Figure_Comparison.png)

### Model Structure

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 10, 64)            64000     
                                                                 
 dropout (Dropout)           (None, 10, 64)            0         
                                                                 
 dense (Dense)               (None, 10, 64)            4160      
                                                                 
 dropout_1 (Dropout)         (None, 10, 64)            0         
                                                                 
 dense_1 (Dense)             (None, 10, 5)             325       
                                                                 
 flatten (Flatten)           (None, 50)                0         
                                                                 
 dense_2 (Dense)             (None, 1)                 51        
                                                                 
=================================================================
```
