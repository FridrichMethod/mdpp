"""Molecule file readers wrapping RDKit supplier classes."""

from __future__ import annotations

import gzip
import logging
import os
from typing import Any, Self

from rdkit import Chem

from mdpp._types import StrPath

logger = logging.getLogger(__name__)


class MolSupplier:
    """Iterate over molecules from a chemical structure file.

    Wraps RDKit's ``MolSupplier`` classes with optional multithreading and
    automatic skipping of empty (unparseable) molecules.

    Recommended for large files to avoid memory issues; use
    ``rdkit.Chem.PandasTools`` for small files and CSV/XLSX formats.

    Examples:
        >>> with MolSupplier("molecules.sdf") as supplier:
        ...     for mol in supplier:
        ...         print(Chem.MolToSmiles(mol))

    Note:
        Molecule ordering is not guaranteed when *multithreaded* is True.
        Use as a context manager (or call :meth:`close`) so any underlying
        gzip / text file handle is released deterministically.
    """

    _THREAD_NUM: int = 0
    _QUEUE_SIZE: int = 1000

    def __init__(self, file: StrPath, *, multithreaded: bool = False, **kwargs: Any) -> None:
        """Initialise the supplier for the given file.

        Args:
            file: Path to the input file (``.sdf``, ``.sdfgz``, ``.mae``,
                ``.maegz``, ``.smi``, or ``.smr``).
            multithreaded: Use multithreaded reading where supported.
                Not available for ``.mae`` / ``.maegz`` files.
            **kwargs: Forwarded to the underlying RDKit supplier.

        Raises:
            TypeError: If the file format is unsupported or multithreading
                is not available for the format.
        """
        ext = os.path.splitext(file)[1].lower()
        # File handles opened here (gzip streams, the .smr text file) that the
        # RDKit supplier does not own; closed by close()/__exit__.
        self._handles: list[Any] = []

        match ext, multithreaded:
            case ".sdf", False:
                self.mol_supplier = Chem.SDMolSupplier(str(file), **kwargs)
            case ".sdf", True:
                self.mol_supplier = Chem.MultithreadedSDMolSupplier(
                    str(file),
                    numWriterThreads=self._THREAD_NUM,
                    sizeInputQueue=self._QUEUE_SIZE,
                    sizeOutputQueue=self._QUEUE_SIZE,
                    **kwargs,
                )
            case ".sdfgz", False:
                fh = gzip.open(file)  # noqa: SIM115 - lifetime tied to close()/__exit__
                self._handles.append(fh)
                # RDKit stubs only document the str overload; the supplier
                # accepts a file-like object at runtime.
                self.mol_supplier = Chem.SDMolSupplier(fh, **kwargs)  # type: ignore[call-overload]
            case ".sdfgz", True:
                fh = gzip.open(file)  # noqa: SIM115 - lifetime tied to close()/__exit__
                self._handles.append(fh)
                self.mol_supplier = Chem.MultithreadedSDMolSupplier(  # type: ignore[call-overload]
                    fh,
                    numWriterThreads=self._THREAD_NUM,
                    sizeInputQueue=self._QUEUE_SIZE,
                    sizeOutputQueue=self._QUEUE_SIZE,
                    **kwargs,
                )
            case ".mae", False:
                self.mol_supplier = Chem.MaeMolSupplier(str(file), **kwargs)
            case ".mae", True:
                raise TypeError("Multithreading is not supported for .mae files.")
            case ".maegz", False:
                fh = gzip.open(file)  # noqa: SIM115 - lifetime tied to close()/__exit__
                self._handles.append(fh)
                self.mol_supplier = Chem.MaeMolSupplier(fh, **kwargs)
            case ".maegz", True:
                raise TypeError("Multithreading is not supported for .maegz files.")
            case ".smi", False:
                self.mol_supplier = Chem.SmilesMolSupplier(str(file), titleLine=False, **kwargs)
            case ".smi", True:
                self.mol_supplier = Chem.MultithreadedSmilesMolSupplier(
                    str(file),
                    titleLine=False,
                    numWriterThreads=self._THREAD_NUM,
                    sizeInputQueue=self._QUEUE_SIZE,
                    sizeOutputQueue=self._QUEUE_SIZE,
                    **kwargs,
                )
            case ".smr", False:
                self._smr_fh = open(file, encoding="utf-8")  # noqa: SIM115
                self._handles.append(self._smr_fh)
                # The supplier attribute is duck-typed across RDKit suppliers
                # and a generator; mypy infers the first branch's type only.
                self.mol_supplier = (  # type: ignore[assignment]
                    Chem.MolFromSmarts(line.strip(), **kwargs) for line in self._smr_fh
                )
            case _:
                raise TypeError(
                    "Unsupported file format. Expected .sdf, .sdfgz, .mae, .maegz, .smi, or .smr."
                )

    def close(self) -> None:
        """Close any file handles opened by this supplier.

        Idempotent: safe to call multiple times. RDKit suppliers constructed
        from a path manage their own handle and are unaffected.
        """
        for handle in self._handles:
            try:
                handle.close()
            except (OSError, ValueError):
                logger.debug("Failed to close a MolSupplier handle.", exc_info=True)
        self._handles = []

    def __enter__(self) -> Self:
        """Enter the runtime context and return the supplier."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close file handles on exit from the runtime context."""
        self.close()

    def __iter__(self) -> Self:
        """Return the iterator."""
        return self

    def __next__(self) -> Chem.rdchem.Mol:
        """Return the next molecule, skipping empty (unparseable) entries."""
        while True:
            if (mol := next(self.mol_supplier)) is not None:
                return mol
            # RDKit stubs claim next() returns a non-None Mol, but at runtime
            # MolSupplier yields None for unparseable entries; the warning
            # branch is reachable.
            logger.warning("Empty molecule is skipped.")  # type: ignore[unreachable]
