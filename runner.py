#!/usr/bin/env python
import os
import toml
import pyarrow as pa
import pyarrow.parquet as pq
from generate_files import Generator


class LCMDataGenJob:
    def __init__(self, config):
        self.config = config
        self.setting = config["job"]["LCMDataGen"]
        self.output_dir_train = self.setting["train_data_dir"]
        self.output_dir_test = self.setting["test_data_dir"]

    def generate_files(self, num_files, output_dir):
        generator = Generator()
        os.makedirs(output_dir, exist_ok=True)
        inputs = list(range(num_files))
        for idx in inputs:
            self.generate_parquet_file(generator, idx, output_dir)

    def generate_parquet_file(self, generator, idx, output_dir):
        fname_prefix = self.setting["file_prefix"]
        fname = f"{fname_prefix}_{idx:04}.parquet"
        fpath = os.path.join(output_dir, fname)

        samples = range(int(self.setting["samples"]))
        table = []
        for _ in samples:
            table.append(generator.generate_row_parquet())
        table = pa.Table.from_pylist(table)
        pq.write_table(table, fpath)
        return idx

    def run(self):
        print("Started creating files")
        self.generate_files(self.setting["num_files_train"], self.output_dir_train)
        print("Done with creating training files")
        self.generate_files(self.setting["num_files_test"], self.output_dir_test)
        print("Done with creating testing files")


if __name__ == "__main__":
    config = toml.load("config.toml")
    a = LCMDataGenJob(config)
    a.run()
