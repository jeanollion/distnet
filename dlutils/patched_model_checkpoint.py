import numpy as np
import warnings
from time import sleep
import subprocess
from keras.callbacks import Callback


class PatchedModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, filepath_dest=None, timeout_function=None, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(PatchedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.filepath_dest=filepath_dest
        self.timeout_function=timeout_function

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def _remove_file(self, filepath):
        removed = False
        while not removed:
            try:
                #print("removing", filepath, "...")
                subprocess.run("rm "+filepath, shell=True, timeout=20)
                removed = True
                #print(filepath, "removed")
            except TimeoutExpired as to:
                if self.timeout_function:
                    print("running timeout function...")
                    self.timeout_function()
                    print("waiting 5 seconds before re-try")
                    sleep(5)
            except Exception as error:
                print("couldn't remove file: ", filepath, "\n", error)

    def _copy_file(self, source, dest):
        self._remove_file(dest)
        copied = False
        while not copied:
            try:
                #print("copying", source, "to", dest, "...")
                subprocess.run("cp "+source+" "+dest, shell=True, timeout=20)
                copied=True
                print(source, "copied to", dest)
            except TimeoutExpired as to:
                if self.timeout_function:
                    print("running timeout function...")
                    self.timeout_function()
                    print("waiting 5 seconds before re-try...")
                    sleep(5)
            except Exception as error:
                print("Couldn't copy {} to {}, error:\n{}".format(source, dest, error))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            filepath_prev = self.filepath.format(epoch=epoch, **logs) if epoch > 0 else None
            filepath_dest = self.filepath_dest.format(epoch=epoch + 1, **logs) if self.filepath_dest else None
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        #self._remove_file(filepath)
                        saved_correctly = False
                        while not saved_correctly:
                            try:
                                if self.save_weights_only:
                                    self.model.save_weights(filepath, overwrite=True)
                                else:
                                    self.model.save(filepath, overwrite=True)
                                if filepath_dest:
                                    self._copy_file(filepath, filepath_dest)
                                if filepath_prev and filepath_prev!=filepath:
                                    self._remove_file(filepath_prev)
                                saved_correctly = True
                            except Exception as error:
                                print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                                sleep(5)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                #self._remove_file(filepath)
                saved_correctly = False
                while not saved_correctly:
                    try:
                        if self.save_weights_only:
                            self.model.save_weights(filepath+str(epoch), overwrite=True)
                        else:
                            self.model.save(filepath+str(epoch), overwrite=True)
                        if filepath_dest:
                            self._copy_file(filepath+str(epoch), filepath_dest)
                        self._remove_file(filepath)
                        if epoch>0:
                            self._remove_file(filepath+str(epoch-1))
                        if not filepath_dest:
                            self._copy_file(filepath+str(epoch), filepath)
                        saved_correctly = True
                    except Exception as error:
                        print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                        sleep(5)
