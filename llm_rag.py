import click
from click_default_group import DefaultGroup
import llm
import sqlite_utils
import sys


@llm.hookimpl
def register_commands(cli):
    @cli.group(
        cls=DefaultGroup,
        default="question",
        default_if_no_args=True,
    )
    def rag():
        "Answer questions against collections using Retrieval Augmented Generation"

    @rag.command()
    @click.argument("question")
    @click.option("collection_name", "-c", "--collection")
    @click.option("-n", default=10, help="Number of documents to consider")
    @click.option(
        "-d",
        "--database",
        type=click.Path(
            file_okay=True, allow_dash=False, dir_okay=False, writable=True
        ),
        envvar="LLM_EMBEDDINGS_DB",
    )
    @click.option("--debug", is_flag=True, help="Show debug context")
    def question(question, collection_name, n, database, debug):
        "Answer a question"
        from llm.cli import get_default_model

        if database:
            db = sqlite_utils.Database(database)
        else:
            db = sqlite_utils.Database(llm.user_dir() / "embeddings.db")
        if "collections" not in db.table_names():
            raise click.ClickException("No collections table found in the database")
        if not collection_name:
            # Use first created collection (TODO: add default collection mechanism)
            options = list(db.query("select name from collections order by id limit 1"))
            if not options:
                raise click.ClickException("No collections found in the database")
            collection_name = options[0]["name"]
        collection = llm.Collection(collection_name, db)
        # Find n similar results to the question
        results = collection.similar(question, number=n)
        with_content = [r for r in results if r.content]
        if not with_content:
            raise click.ClickException(
                "No results with content found for that question"
            )
        context = "\n".join([r.content for r in with_content])
        # Run the prompt
        model_id = get_default_model()
        model = llm.get_model(model_id)
        if model.needs_key:
            model.key = llm.get_key(None, model.needs_key, model.key_env_var)

        response = model.prompt(
            context,
            system="You answer questions based on the provided context. Answer this question: {}".format(
                question
            ),
        )
        for chunk in response:
            print(chunk, end="")
            sys.stdout.flush()
        print("")
        if debug:
            print("----\nContext:")
            print(context)
            print("Used model: {}".format(model.model_id))

    @rag.command()
    @click.argument("collections", required=False, nargs=-1)
    def default_collection(collections):
        "View or set the default collection"
        if collections:
            click.echo("Setting default collection: {}".format(collections))
        else:
            click.echo("Default collection is X")

    @rag.command()
    @click.argument("model", required=False)
    @click.option("-c", "--collection")
    def default_model(model, collection):
        "View or set the default model for RAG"
        if model:
            click.echo("Setting default model: {}".format(model))
            if collection:
                click.echo("  For collection: {}".format(collection))
        else:
            click.echo("Default model is Y")

    # TODO: configure_collection perhaps, for setting other stuff?
