import caption_generator
from keras.callbacks import ModelCheckpoint

def train_model(weight = None, batch_size=32, epochs = 10):

    cg = caption_generator.CaptionGenerator()
    model = cg.create_model()

    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = 'weights-improvement-epoch-{epoch:02d}-val_acc-{val_acc:.5f}-val_loss-{val_loss:.5f}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(cg.data_generator(path = 'Flickr8k_text/flickr_8k_train_dataset.txt',batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=epochs, verbose=2, callbacks=callbacks_list,validation_steps=cg.total_samples_dev/batch_size,validation_data=cg.data_generator(path = 'Flickr8k_text/flickr_8k_dev_dataset.txt',batch_size=batch_size))
    try:
        model.save('Models/WholeModel.h5', overwrite=True)
        model.save_weights('Models/Weights.h5',overwrite=True)
    except:
        print "Error in saving model."
    print "Training complete...\n"

if __name__ == '__main__':
    train_model(epochs=100)

