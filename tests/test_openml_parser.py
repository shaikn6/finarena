"""Guard for `make train`: scripts/train_credit.py loads data via fetch_openml(parser="auto"), i.e. scikit-learn's
pandas ARFF parser. Under pandas 3 that parser crashes on numeric nominal labels such as the {0,1} target."""
import gzip
import io

from sklearn.datasets._arff_parser import load_arff_from_gzip_file


def test_sklearn_arff_parser_reads_numeric_nominal_target():
    arff = b"@RELATION t\n@ATTRIBUTE x1 NUMERIC\n@ATTRIBUTE y {0,1}\n@DATA\n1.5,0\n2.5,1\n"
    columns = {"x1": {"data_type": "numeric"}, "y": {"data_type": "nominal"}}
    X, y, _, _ = load_arff_from_gzip_file(
        gzip.GzipFile(fileobj=io.BytesIO(gzip.compress(arff))), parser="pandas", output_type="pandas",
        openml_columns_info=columns, feature_names_to_select=["x1"], target_names_to_select=["y"])
    assert X["x1"].tolist() == [1.5, 2.5]
    assert y.astype(int).tolist() == [0, 1]
