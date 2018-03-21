

# Learning from Artificial Neural Networks

Pretrained Artificial Neural Networks used to work like a Blackbox: You hand them an input and they predict an output with a certain probability – but without us knowing the internal processes of how they came up with their prediction. A Neural Network to recognize images usually consists of around 20 neuron layers, trained with millions of images to tweak the network parameters to give high quality classifications.

The layers consist of neurons that are trained to only forward information if they recognize one specific image feature, resulting in an action potential that serves as an input for the neurons of the next deeper layer. Each layer gets the information of the previous layer and supplies information to the next one until the output layer states the networks prediction. How many neurons of a certain layer fired their action potential implies how strongly the layer recognized its training features in the provided image.

 ![http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/](https://github.com/joelbarmettlerUZH/DeepDream_Intro/raw/master/Resources/1_NeuralNetwork_Model.png)

*Figure  1: Structure of a simple, 2 layer Neural Network. [Source](http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/)*

One of the little things we know about the functionality of Neural Network that recognize images is that each additional layer extracts higher level features of the image: While the first layer looks for edges and corners, the middle layers recognize shapes, the last layers whole objects and compositions. The human visual system follows a quiet similar approach: Rods in the retina forward action potential on contrasts and edges, the visual cortex unites information to more concrete objects. With the basic knowledge of what individual layers are roughly doing, a technique is needed to gather insights about an already trained network.

 ![https://distill.pub/2017/feature-visualization/](https://github.com/joelbarmettlerUZH/DeepDream_Intro/raw/master/Resources/2_Layer_Representation.png)

*Figure 2: What a neuron / layer has learned to recognize. [Source](https://distill.pub/2017/feature-visualization/)*

To understand the ongoing processes inside the network, researches focus on feature-visualization and attribution. While feature visualization tries to find out what parts of the Neural Network are trained on which image features by generating example images from a pretrained network, attribution wants to find out what parts of the input images were responsible to lead the Neural Network to its output prediction. By applying feature visualization on the Neural Network, we can start to understand the different parts of the Neural Network and eventually conclude our findings to the human brain as well.

As a simple approach, we can show a pretrained Neural Network random noise as input image and measure the output of the network layer of which we want to extract insight information. With manipulating the image towards higher action potentials, we can find out what sort of input images deliver the highest action potential – implying that the layer was trained to recognize exactly the features we have created. As an example, we could examine the second layer of an image recognition Neural Network, responsible for shapes and contrast, by measuring the action potential it creates. When showing the Neural Network random noise, the measured action potential happens to be quite small since nearly no features are recognized. With adding edges, colours and shapes to the noise, the second layer of the Neural Network will increase its potential, recognizing the structures it is trained for. By manipulating the image and trying to get a higher action potential, we can eventually find the optimal image for the layer, being left with an image that exactly represents what the layer is looking for.

 ![https://distill.pub/2017/feature-visualization/](https://github.com/joelbarmettlerUZH/DeepDream_Intro/raw/master/Resources/3_Noise_to_Representation.png)

*Figure  3: Manipulating noise to get optimized image. [Source](https://distill.pub/2017/feature-visualization/)*

Instead of manually manipulating an image, we could automatize the optimization process by inversing the Neural Network: The activation functions of the pretrained network can be inversed to not extract information from an image but insert it back into the image. With providing the inversed Neural Network a random noise image, it inserts the features it is trained for and delivers an optimized image that represents the optimal, best fitting training image.

Such an optimized training image can provide powerful information about what kind of images the Neural Network will recognize. While the input dataset may be pictures of dumbbells, the Neural Network may have trained itself to focus on the hand holding the weight as well – leaving to potentially unrecognized pure dumbbells or false-positives on hands.  During the training phase of a Neural Network, we could generate such optimization images and check whether the layers still behave as we expect them to.

![https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html](https://github.com/joelbarmettlerUZH/DeepDream_Intro/raw/master/Resources/4_dumbbell_optimized.png)

*Figure  4: Mistraining of dumbbells including arms. [Source](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)*

We can also feed the Neural Network a real image and add the output of a certain layer on top of the original image, leaving us with an image that shows us what a pretrained Neural Network has recognized in this image. Assume a Neural Network was trained to recognize animals: When feeding the Neural Network an image of an antelope and letting it enhance the second layer, capable of recognizing lines, we are left with the same image of an antelope but with enhanced lines.

 ![https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html](https://github.com/joelbarmettlerUZH/DeepDream_Intro/raw/master/Resources/5_antelope_enhanced.jpg)

*Figure  5: Enhancing first Layers on an Image. [Source](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)*

Things get really interesting when we apply this practice to images that do not fit the training data of the Neural Network: When we show a Neural Network that was trained to recognize buildings an image of a tree and ask it to enhance one of the last layers, capable of recognizing objects and compositions, we are left with a strange interpretation of the tree as a building.

 ![https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html](https://github.com/joelbarmettlerUZH/DeepDream_Intro/raw/master/Resources/6_building_tree.png)

*Figure  6: Enhance buildings in an image of a tree. [Source](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)*

Such games can be brought to the extreme by iteratively enhancing the output image again and again, creating images that look like they come directly from an acid trip. The first model that created such output images was made by google and named Deep Dream.

 ![https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html](https://github.com/joelbarmettlerUZH/DeepDream_Intro/raw/master/Resources/7_deepdream.png)

*Figure  7: Enhance animals &amp; cars in a Landscape. [Source](https://www.theatlantic.com/technology/archive/2015/09/robots-hallucinate-dream/403498/)*

It turns out that the similarity between acid trips and enhanced Neural Network layer is scientifically measurable: When taking drugs like LSD, the brain overstimulates certain regions in the visual cortex, leading to the generability of patterns as we see them in Deep Dreamed images. What we perceive is then a mixture of what we see and what we are expecting to see. Applied to the Neural Network: Seeing an image with over-represented layers and expecting (_being trained_) to see cars and animals.  Applied to the Human Neural System: Making the brain reimagine what it sees to cause neurons in particular layers of the visual cortex to fire more and more.

But Deep Dream seems to have more impact on science than just modelling LSD trips. Researchers suggest that Deep Dream may model different psychotic phenomena such as aberrant salience, a mechanism suggested for pathogenesis of psychosis that makes it hard to recognize objects, very accurately. Meanwhile, Neural Networks that simulate schizophrenia are being modelled to later examine what neural manipulation was needed to cause the schizophrenic effect. Studies of Artificial Neural Network may illuminate many aspects of brain function in health and disease, as well as dreams and creativity. Increasing understanding of the brain&#39;s circuit dynamics and connectomic alterations in neuropsychiatric disorders make studies of Neural Network models of these illnesses increasingly timely.

References
----

- [Inceptionism: Going Deeper into Neural Networks ](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
- [Feature Visualization - How neural networks build up their understanding of images](https://distill.pub/2017/feature-visualization/)
- [When Robots Hallucinate](https://www.theatlantic.com/technology/archive/2015/09/robots-hallucinate-dream/403498/)
- [Deep dreaming, aberrant salience and psychosis: Connecting the dots by artificial neural networks](http://www.schres-journal.com/article/S0920-9964(17)30029-4/pdf)


License
----

License is only grated on the self-written content, not on the resources taken from the reference papers.

MIT License

Copyright (c) 2018 Joel Barmettler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
