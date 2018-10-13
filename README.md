
Create python package:

python setup.py sdist --formats=gztar

Upload the distribution to GCS:

gsutil cp dist/sentiment-analyzer-0.1.tar.gz your-bucket-name/package/latest.tar.gz

submit training job:

gcloud ml-engine jobs submit training newsgroup_classification_1
--region europe-west1
--runtime-version 1.6
--config config.yml
--package-path trainer
--module-name trainer.task
--job-dir your-bucket-name/jobdir/
--packages your-bucket-name/package/latest.tar.gz
--
--x_train_file your-bucket-name/data/x_train.csv
--y_train_file your-bucket-name/data/y_train.csv
--x_test_file your-bucket-name/data/x_test.csv
--y_test_file your-bucket-name/data/y_test.csv
--number_of_epochs 500
--number_of_features 300
--number_of_training_examples 2034
--number_of_test_examples 1353
--batch_size 2000
--model_location newsgroup_classification_1

submit training job with hyperparameter tuning:

gcloud ml-engine jobs submit training hyperparameter_tuning_2
--region europe-west1
--runtime-version 1.6
--config hyperparameter_tuning_config.yml
--package-path trainer
--module-name trainer.hyperparameter_tuning_task
--job-dir your-bucket-name/jobdir/
--packages your-bucket-name/package/latest.tar.gz
--
--x_train_file your-bucket-name/data/x_train.csv
--y_train_file your-bucket-name/data/y_train.csv
--x_test_file your-bucket-name/data/x_test.csv
--y_test_file your-bucket-name/data/y_test.csv
--number_of_epochs 200
--number_of_features 300
--number_of_training_examples 2034
--number_of_test_examples 1353
--batch_size 2000
--model_location hyperparameter_tuning_2
--number_of_nodes_one 10
--number_of_nodes_two 10
