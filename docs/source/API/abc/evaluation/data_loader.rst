dataset loader
==============

.. autoclass:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader

Methods
-------

.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader.available_dataset_names
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader.available_splits
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader.check_dataset_names
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader.check_splits
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader.load_corpus
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader.load_qrels
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader.load_queries
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._load_remote_corpus
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._load_remote_qrels
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._load_remote_queries
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._load_local_corpus
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._load_local_qrels
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._load_local_queries
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._download_file
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._get_fpath_size
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._download_gz_file
.. automethod:: FlagEmbedding_Aizip.abc.evaluation.AbsEvalDataLoader._download_zip_file