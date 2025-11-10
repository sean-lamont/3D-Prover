from __future__ import division, absolute_import, print_function

from omegaconf import OmegaConf

'''
Generator util to iterate through a MongoDB cursor with a given batch size
'''


def get_batches(cursor, batch_size):
    batch = []
    for i, row in enumerate(cursor):
        if i % batch_size == 0 and i > 0:
            yield batch
            del batch[:]
        batch.append(row)
    yield batch


def config_to_dict(conf):
    return OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )
