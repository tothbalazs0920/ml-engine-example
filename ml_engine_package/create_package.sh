### Build and upload the package ###

# Delete existing distribution
rm -f MANIFEST
rm -rf dist
rm -rf sentiment_analyzer.egg-info

# Build the distribution
python setup.py sdist --formats=gztar
# Upload the distribution to GCS
gsutil cp dist/sentiment-analyzer-0.1.tar.gz gs://20-percent-mlengineexample/package/latest.tar.gz

echo "Distribution uploaded"