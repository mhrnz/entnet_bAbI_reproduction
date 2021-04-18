import numpy as np
import tensorflow as tf
import pickle
from tensorflow.contrib import seq2seq
from keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard


def calculate_loss(targets, outputs):
    """
    Args:
        inputs: outputs shape : [batch_size,max_sents_num*max_sents_len, vocab_size]
                lstm_targets shape : [batch_size, max_sents_num*max_sents_len]
                mask : [batch_size, max_sents_num*max_sents_len]

    """

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,labels=targets)
    return loss


def train(prgrphs, prgrphs_mask, questions, answers, keys, keys_mask,  embedding_matrix, max_entity_num,
          entity_embedding_dim, vocab_size, learning_rate, save_path, batch_size, validation_split, epochs):



      def schedule(epoch_num, lr):
        new_lr=lr
        if epoch_num!=0 and epoch_num%25==0:
          new_lr=lr/2
        return new_lr

      num_batches=((1-validation_split)*prgrphs.shape[0])//batch_size
      train_loss=[]
      train_acc=[]
      val_loss=[]
      val_acc=[]

      def save_loss(batch,logs):
        nonlocal train_loss
        nonlocal train_acc
        if batch==0:
          train_loss=[]
          train_acc=[]
        train_loss.append(logs['loss'])
        train_acc.append(logs['acc'])

      def plot(loss, acc, rangee, mode, epoch=200):
          fig=plt.figure()
          ax1=fig.add_subplot(211)
          ax1.plot(np.arange(rangee),loss)
          ax1.set_xlabel(mode+" "+str(epoch))
          ax1.set_ylabel("loss")
          ax2=fig.add_subplot(212)
          ax2.plot(np.arange(rangee),acc)
          ax2.set_ylabel("accuracy")
          plt.show()


      start_time=time.time()
      def plot_train_loss(epoch,logs):
        nonlocal start_time
        end_time=time.time()
        print("time:",end_time-start_time)
        start_time=end_time
        val_loss.append(logs['val_loss'])
        val_acc.append(logs['val_acc'])
        plot(train_loss,train_acc,num_batches,'train',epoch)
        
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      session = tf.Session(config=config)
      tf.keras.backend.set_session(session)
      
      model = Model(embedding_matrix=embedding_matrix, vocab_size=vocab_size, max_entity_num=max_entity_num,
                        entity_embedding_dim=entity_embedding_dim)
      print('leraning_rate',learning_rate)
      adam = tf.keras.optimizers.Adam(lr=learning_rate)
      model.compile(optimizer=adam,
                    loss=calculate_loss,
                    clip_gradients=0.05,
                    metrics=['accuracy'])

      cp_callback = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                       save_weights_only=True,
                                                       verbose=1)    

      tn_callback=tf.keras.callbacks.TerminateOnNaN()
      lr_callback=tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
      lambda_callback=tf.keras.callbacks.LambdaCallback(on_batch_end=save_loss, on_epoch_end=plot_train_loss)

      answerss=np.zeros([answers.shape[0],vocab_size],np.int32)
      answerss[np.arange(answers.shape[0]),np.squeeze(answers)]=1

      history = model.fit(x=[prgrphs, prgrphs_mask, questions, keys, keys_mask], y=answerss, batch_size=batch_size,
                          validation_split=validation_split, epochs=epochs, 
                          callbacks=[cp_callback, lambda_callback,tn_callback, lr_callback],shuffle=True)
      
      plot(val_loss,val_acc,epochs,'validation')
      return history

