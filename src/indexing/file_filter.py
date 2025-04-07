import os


def make_file_filter(config):
    include_extensions = set(config.get("include_extensions", []))
    exclude_extensions = set(config.get("exclude_extensions", []))
    exclude_dirs = set(config.get("exclude_dirs", []))

    def file_filter(file_path: str) -> bool:
        """
        This filter allows specifying which file extensions to include or exclude.
        If include_extensions is not provided, all extensions are allowed by default.
        """

        file_path = os.path.join(*file_path.split(os.sep)[2:])
        file_extension = "." + file_path.split(".")[-1] if "." in file_path else ""

        if any(file_path.startswith(d) for d in exclude_dirs):
            return False
        if any(file_path.endswith(ext) for ext in exclude_extensions):
            return False
        if include_extensions:
            return file_extension in include_extensions
        return True

    return file_filter
