export PYTHONPATH=/ace:$PYTHONPATH
pytest -sv datasets/dataset2ace.py::test_vlp2ace
# pytest -sv datasets/dataset2ace.py::test_colmap2ace
