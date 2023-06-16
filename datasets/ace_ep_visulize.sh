export PYTHONPATH=/ace:$PYTHONPATH
pytest -sv datasets/dataset2ace.py::test_ace_visulize
pytest -sv datasets/dataset2ace.py::test_ace_visulize_vlp
pytest -sv datasets/dataset2ace.py::test_ace_visulize_colmap
