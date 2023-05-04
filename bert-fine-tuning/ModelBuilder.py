import tensorflow as tf


class ModelBuilder:

    def __init__(self, optimizer, loss_fn, **kwargs):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
