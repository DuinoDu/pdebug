"""Runner for pipeline."""
import logging
from abc import ABC, abstractmethod
from collections import Counter
from itertools import chain
from typing import Any, Dict, Iterable

from pdebug.pdag.io import AbstractDataSet, DataCatalog, MemoryDataSet
from pdebug.pdag.pipeline import Pipeline
from pdebug.pdag.pipeline.node import Node


class AbstractRunner(ABC):
    """``AbstractRunner`` is the base class for all ``Pipeline`` runner
    implementations.
    """

    @property
    def _logger(self):
        return logging.getLogger(self.__module__)

    def run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        session_id: str = None,
    ) -> Dict[str, Any]:
        """Run the ``Pipeline`` using the datasets provided by ``catalog``
        and save results back to the same objects.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            session_id: The id of the session.

        Raises:
            ValueError: Raised when ``Pipeline`` inputs cannot be satisfied.

        Returns:
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.

        """

        catalog = catalog.shallow_copy()

        unsatisfied = pipeline.inputs() - set(catalog.list())
        # import ipdb; ipdb.set_trace()
        if unsatisfied:
            raise ValueError(
                f"Pipeline input(s) {unsatisfied} not found in the DataCatalog"
            )

        free_outputs = pipeline.outputs() - set(catalog.list())
        unregistered_ds = pipeline.data_sets() - set(catalog.list())
        for ds_name in unregistered_ds:
            catalog.add(ds_name, self.create_default_data_set(ds_name))

        self._run(pipeline, catalog, session_id)

        # self._logger.debug("Pipeline execution completed successfully.")

        return {ds_name: catalog.load(ds_name) for ds_name in free_outputs}

    @abstractmethod  # pragma: no cover
    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        session_id: str = None,
    ) -> None:
        """The abstract interface for running pipelines, assuming that the
        inputs have already been checked and normalized by run().

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            session_id: The id of the session.

        """

    @abstractmethod  # pragma: no cover
    def create_default_data_set(self, ds_name: str) -> AbstractDataSet:
        """Factory method for creating the default dataset for the runner.

        Args:
            ds_name: Name of the missing dataset.

        Returns:
            An instance of an implementation of ``AbstractDataSet`` to be
            used for all unregistered datasets.

        """

    def _suggest_resume_scenario(
        self, pipeline: Pipeline, done_nodes: Iterable[Node]
    ) -> None:
        remaining_nodes = set(pipeline.nodes) - set(done_nodes)

        postfix = ""
        if done_nodes:
            node_names = (n.name for n in remaining_nodes)
            resume_p = pipeline.only_nodes(*node_names)

            start_p = resume_p.only_nodes_with_inputs(*resume_p.inputs())
            start_node_names = (n.name for n in start_p.nodes)
            postfix += f"  --from-nodes \"{','.join(start_node_names)}\""

        # self._logger.warning(
        #     "There are %d nodes that have not run.\n"
        #     "You can resume the pipeline run by adding the following "
        #     "argument to your previous command:\n%s",
        #     len(remaining_nodes),
        #     postfix,
        # )


def run_node(
    node: Node,
    catalog: DataCatalog,
    session_id: str = None,
) -> Node:
    """Run a single `Node` with inputs from and outputs to the `catalog`.

    Args:
        node: The ``Node`` to run.
        catalog: A ``DataCatalog`` containing the node's inputs and outputs.
        session_id: The session id of the pipeline run.

    Returns:
        The node argument.

    """
    inputs = {}
    for name in node.inputs:
        inputs[name] = catalog.load(name)

    # __import__('ipdb').set_trace()

    # run node
    outputs = node.run(inputs)

    for name, data in outputs.items():
        if data is None:
            print(f"{node} please provide output, rather than return None.")
        catalog.save(name, data)

    for name in node.confirms:
        catalog.confirm(name)
    return node


class SequentialRunner(AbstractRunner):
    """``SequentialRunner`` is an ``AbstractRunner`` implementation. It can
    be used to run the ``Pipeline`` in a sequential manner using a
    topological sort of provided nodes.
    """

    def create_default_data_set(self, ds_name: str) -> AbstractDataSet:
        """Factory method for creating the default data set for the runner.

        Args:
            ds_name: Name of the missing data set

        Returns:
            An instance of an implementation of AbstractDataSet to be used
            for all unregistered data sets.

        """
        return MemoryDataSet()

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        session_id: str = None,
    ) -> None:
        """The method implementing sequential pipeline running.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            session_id: The id of the session.

        Raises:
            Exception: in case of any downstream node failure.
        """
        nodes = pipeline.nodes
        done_nodes = set()

        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))

        for exec_index, node in enumerate(nodes):
            try:
                print("\x1b[6;30;42m" + f"[{node.name}]" + "\x1b[0m")
                run_node(node, catalog, session_id)
                done_nodes.add(node)
            except Exception:
                self._suggest_resume_scenario(pipeline, done_nodes)
                raise

            # decrement load counts and release any data sets we've finished with
            for data_set in node.inputs:
                load_counts[data_set] -= 1
                if (
                    load_counts[data_set] < 1
                    and data_set not in pipeline.inputs()
                ):
                    catalog.release(data_set)
            for data_set in node.outputs:
                if (
                    load_counts[data_set] < 1
                    and data_set not in pipeline.outputs()
                ):
                    catalog.release(data_set)

            # self._logger.debug(
            #     "Completed %d out of %d tasks", exec_index + 1, len(nodes)
            # )
