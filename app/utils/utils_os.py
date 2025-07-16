import glob
import os
from itertools import chain


class OsCls:
    @staticmethod
    def get_base_path():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    @staticmethod
    def get_current_path():
        return os.getcwd()

    @classmethod
    def get_import_path(cls, path=None):
        import_path = os.path.join(cls.get_base_path(), "_import")
        result_path = os.path.join(import_path, path) if path else import_path
        cls.create_path_if_not_exist(result_path)
        return result_path

    @classmethod
    def get_import_cognos_path(cls):
        import_cognos_path = os.path.join(cls.get_import_path(), "cognos")
        cls.create_path_if_not_exist(import_cognos_path)
        return import_cognos_path

    @classmethod
    def get_import_archive_path(cls):
        import_archive_path = os.path.join(cls.get_import_path(), "archive")
        cls.create_path_if_not_exist(import_archive_path)
        return import_archive_path

    @classmethod
    def get_import_bad_path(cls):
        import_bad_path = os.path.join(cls.get_import_path(), "bad")
        cls.create_path_if_not_exist(import_bad_path)
        return import_bad_path

    @staticmethod
    def get_files_list(path, file_masks="*.xlsx"):
        """Return the list of directory files by masks.
        Masks can be: single str, tuple or list of str.
        Examples of masks: '*.xlsx', ('*.xlsx', '*.csv'),..."""
        if isinstance(file_masks, str):
            file_masks = [file_masks]
        return set(chain(*(glob.glob(os.path.join(path, mask)) for mask in file_masks)))

    @staticmethod
    def get_username():
        userhome = os.path.expanduser("~")  # Gives user's home directory
        return os.path.split(userhome)[-1]  # Gives username by splitting path based on OS

    @staticmethod
    def join_path(*args):
        return os.path.join(*args)

    @staticmethod
    def create_path_if_not_exist(path):
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except OSError as e:
                print(f"Failed to create {path} directory (Error: {e}")
            else:
                print(f"The directory {path} was successfully created")
