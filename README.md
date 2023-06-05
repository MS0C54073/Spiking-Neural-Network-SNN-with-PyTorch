# Spiking-Neural-Network-SNN-with-PyTorch
#РУССКИЙ
Создана модель для проведения сравнения, не стремясь превзойти какие-либо стандарты. Следует отметить, что результаты достаточно похожи, хотя Сверхпроводящая Нейронная Сеть (SNN) кажется проявляет немного более высокую производительность, хотя и с более длительным временем обучения.

Также эксперимент показывает, что минимальная настройка гиперпараметров была проведена до сих пор, поскольку код был написан и опубликован сразу. В результате существует потенциал для значительных вариаций производительности при дальнейших настройках.

Сверхпроводящие нейронные сети могут служить техникой регуляризации, подобной отсеву (dropout), так как ожидается, что нейроны будут стрелять не одновременно. Учитывая роль ритмов мозга, которые не происходят в традиционных подходах глубокого обучения.

Кроме того, существует возможная связь между обратным распространением и обучением Хебба в Глубоких Нейронных Сетях (DNN), включающая в себя включение оси времени и поведение угасания и возбуждения.

"Как это работает?" - спросил собеседник.

Конечно, давайте погрузимся в детали.

Метод активации нейрона включает в себя следующие шаги, где аргумент входа обозначается как 'x':

Во-первых, важно инициализировать (или очистить) состояние каждого нейрона при начале предсказаний. Для этого создаются тензоры для 'prev_inner' и 'prev_outer', оба установлены в ноль, отрегулированы для размера пакета и размерностей скрытого слоя, и выделены соответствующему устройству:

self.prev_inner = torch.zeros([batch_size, self.n_hidden]).to(self.device)
self.prev_outer = torch.zeros([batch_size, self.n_hidden]).to(self.device)

Затем вход 'x', представляющий рукописные цифры MNIST, умножается на матрицу весов с использованием функции 'fully_connected':

input_excitation = self.fully_connected(x)

Полученное значение затем прибавляется к затухающей версии внутренней активации нейрона с предыдущего шага времени или временного шага (Δt - прошедшее время). Это затухание контролируется 'decay_multiplier', который постепенно уменьшает внутреннюю активацию, чтобы предотвратить избыточное накопление и обеспечить покой нейрона. Это можно продемонстрировать, используя decay_multiplier равный 0.9. Эта форма затухания, известная как экспоненциальное затухание, вводит эффект, подобный экспоненциальному скользящему среднему со временем, который влияет на градиенты во время обратного распространения. Повторным умножением внутренней активации на 0.9 со временем она затухает, позволяя нейронам снимать возбуждение перед стрельбой. Следовательно, утверждение "нейроны, стреляющие вместе, связаны вместе" становится более точным: когда пресинаптический вход получен ближе к моменту генерации выхода, его недавнее значение не будет иметь существенного затухания. Это усиливает градиенты нейронов, которые участвуют в возбуждении текущего стреляющего нейрона, обеспечивая эффективное обучение с помощью градиентного спуска. Напротив, стимулы, произошедшие слишком давно, будут иметь затухающие градиенты из-за экспоненциального затухания, что делает их менее полезными для процесса обучения во время обратного распространения, следуя принципам обучения Хебба:

inner_excitation = input_excitation + self.prev_inner * self.decay_multiplier

Следующий шаг заключается в вычислении активации нейронов для определения их выходного значения. Порог должен быть достигнут, прежде чем нейрон активируется. Хотя функция ReLU может не быть наиболее подходящим выбором здесь (об этом позже), она была реализована для быстрого разработки рабочего прототипа:

outer_excitation = F.relu(inner_excitation - self.threshold)

Теперь интересная часть. Если нейрон стреляет, его активация вычитается из его внутреннего состояния для сброса нейрона. Это действие служит двум целям: во-первых, оно возвращает нейрон в покоящееся положение, предотвращая постоянную стрельбу после активации; во-вторых, оно гарантирует, что каждое событие стрельбы изолировано друг от друга, эффективно обрезая градиенты по времени. Сверхпроводящие нейронные сети (SNN) черпают вдохновение из мозга, где естественные нейроны также имеют период отдыха, требующий кратковременной паузы перед повторной стрельбой, даже если они полностью возбуждены нижними нейронами, служащими в качестве входов. Чтобы имитировать этот период покоя, происходит второй штраф, называемый 'penalty_threshold', который вычитается сразу после сброса порога. Следует отметить, что точное место нахождения отрицательной части в биологическом периоде покоя (например, аксон против тела) неизвестно. Чтобы упростить, отрицательное наказание применяется внутри нейронов. В следующем фрагменте кода демонстрируется вычитание штрафа только тогда, когда нейрон стреляет:

do_penalize_gate = (outer_excitation > 0).float()
inner_excitation = inner_excitation - (self.penalty_threshold + outer_excitation) * do_penalize_gate

Наконец, предыдущий вывод возвращается, имитируя небольшую задержку перед стрельбой. Хотя в настоящее время это избыточная функция, она может оказаться полезной, если SNN будет включать рекуррентные связи, требующие временных сдвигов в связях от верхних слоев около выходов к нижним слоям около входов. Обновляются предыдущие внутреннее и внешнее состояния, и возвращаются отложенное состояние и вывод:

delayed_return_state = self.prev_inner
delayed_return_output = self.prev_outer
self.prev_inner = inner_excitation
self.prev_outer = outer_excitation
return delayed_return_state, delayed_return_output

Для выполнения классификации значения выходных спайковых нейронов классификации усредняются по оси времени, чтобы получить одно число для каждого класса. Эти усредненные значения затем проходят через функцию потерь softmax cross-entropy для классификации и последующего обратного распространения ошибки:

output = torch.mean(output, dim=0)
loss = F.cross_entropy(output, target)
loss.backward()

Это, вкратце, описывает основную логику работы сверхпроводящих нейронных сетей (SNN). Разумеется, в реальном коде могут быть дополнительные детали и оптимизации, но эти шаги составляют основу функционирования SNN.

Надеюсь, это помогло разъяснить, как работает сверхпроводящая нейронная сеть (SNN). Если у вас возникнут дополнительные вопросы, не стесняйтесь задавать!
*******************
#ENGLISH
Trained a model to make a comparison, without aiming to exceed any benchmarks. Note that the results are relatively similar, although the Spiking Neural Network (SNN) appears to exhibit slightly better performance, albeit with a longer training time.

Also the experiment clarifies that minimal hyperparameter tuning has been conducted thus far, as the code was written and shared in one go. Consequently, there is potential for significant performance variations with further adjustments. 

SNNs can serve as a regularization technique, akin to dropout, as the firing of neurons is not expected to occur simultaneously. Given the role of brain rhythms in the brain, which do not occur in traditional deep learning approaches.

Additionally, there is a possible connection between backpropagation and Hebbian learning in Deep Neural Networks (DNNs), involving the incorporation of a time axis and refractory firing behavior.


"How does it work?" inquired the individual.

Certainly, let's delve into the intricacies.

The firing method for defining a neuron involves the following steps, with the input argument denoted as 'x':

Firstly, it is essential to initialize (or empty) the state for each neuron when commencing predictions. This entails creating tensors for 'prev_inner' and 'prev_outer', both set to zero, adjusted for the batch size and hidden layer dimensions, and allocated to the appropriate device:

self.prev_inner = torch.zeros([batch_size, self.n_hidden]).to(self.device)
self.prev_outer = torch.zeros([batch_size, self.n_hidden]).to(self.device)

Next, the input 'x', which represents handwritten MNIST digits, 
undergoes multiplication with a weight matrix using the 'fully_connected' function:

input_excitation = self.fully_connected(x)

The resulting value is then added to a decayed version of the neuron's inner activation from the previous time step or time tick 
(Δt time elapsed). This decay is controlled by the 'decay_multiplier,' which gradually diminishes the inner activation to prevent excessive accumulation and ensure neuron rest. It can be exemplified by using a decay_multiplier of 0.9. This form of decay, known as exponential decay, introduces an effect similar to an exponential moving average over time, which influences the gradients during backpropagation. By repeatedly multiplying the inner activation by 0.9 over time, it decays, allowing neurons to unexcite themselves before firing. Consequently, the statement "neurons that fire together wire together" becomes more accurate: when a pre-synaptic input is received closer to the moment of generating an output, its recent value will not have undergone significant decay. This strengthens the gradients of the neurons involved in exciting the currently firing neuron, enabling effective learning through gradient descent. Conversely, stimuli that occurred too long ago will experience vanishing gradients due to exponential decay, rendering them less useful for the learning process during backpropagation, thus adhering to the principles of Hebbian learning:

inner_excitation = input_excitation + self.prev_inner * self.decay_multiplier


The subsequent step involves computing the activation of the neurons to determine their output value. A threshold must be reached before a neuron activates. While the ReLU function may not be the most appropriate choice here (more on that later), it was implemented to quickly develop a working prototype:

outer_excitation = F.relu(inner_excitation - self.threshold)


Now comes the intriguing part. If a neuron fires, its activation is subtracted from its inner state to reset the neuron. This action serves two purposes: first, it resets the neuron to a resting position, preventing constant firing after activation; second, it ensures that each firing event is isolated from one another, effectively clipping the gradient through time. Spiking Neural Networks (SNNs) draw inspiration from the brain, where natural neurons also exhibit a refractory period, requiring a brief pause before firing again, even if fully excited by lower neurons that serve as inputs. To simulate this refractory period, a second penalty, referred to as 'penalty_threshold,' is subtracted immediately after resetting the threshold. It should be noted that the exact location of the negative part in the biological refractory period (e.g., axon versus body) is uncertain. To keep things simple, the negative penalty is applied within the neurons. The following code snippet demonstrates the subtraction of the penalty only when the neuron fires:

do_penalize_gate = (outer_excitation > 0).float()
inner_excitation = inner_excitation - (self.penalty_threshold + outer_excitation) * do_penalize_gate


Finally, the previous output is returned, simulating a slight firing delay. Although currently redundant, this feature could prove useful if the SNN were to include recurrent connections that require time offsets in the connections from top layers near the outputs back to bottom layers near the inputs. The previous inner and outer states are updated, and the delayed return state and output are returned:

delayed_return_state = self.prev_inner
delayed_return_output = self.prev_outer
self.prev_inner = inner_excitation
self.prev_outer = outer_excitation
return delayed_return_state, delayed_return_output


To perform classification, the values of the classification output spiking neurons are averaged over the time axis to obtain one number per class. These averaged values are then passed through the softmax cross-entropy loss function for classification and subsequent backpropagation. Consequently, the present SNN PyTorch class can be reused within any other feedforward neural network, as it repeats inputs over time with random noisy masks and averages outputs over time.

Remarkably, it worked on the first attempt once the dimension mismatch errors were resolved. The accuracy achieved was similar to that of a simple non-spiking Feedforward Neural Network with the same number of neurons, even without fine-tuning the threshold. Ultimately, I realized that coding and training a Spiking Neural Network (SNN) with PyTorch is relatively straightforward, as demonstrated above. It can be implemented in a single evening.

In essence, the activation of neurons must decay over time and fire only when surpassing a certain threshold. Thus, the output of the results
