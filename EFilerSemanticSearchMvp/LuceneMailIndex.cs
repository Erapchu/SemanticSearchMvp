using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.QueryParsers.Classic;
using Lucene.Net.Search;
using Lucene.Net.Store;
using Lucene.Net.Util;

namespace EFilerSemanticSearchMvp;

public sealed class LuceneMailIndex : IDisposable
{
    private const LuceneVersion Version = LuceneVersion.LUCENE_48;
    private readonly RAMDirectory _directory = new();
    private readonly StandardAnalyzer _analyzer = new(Version);

    public void Build(IEnumerable<MailMessage> messages)
    {
        var config = new IndexWriterConfig(Version, _analyzer);
        using var writer = new IndexWriter(_directory, config);

        foreach (var mail in messages)
        {
            var doc = new Document
            {
                new StringField("id", mail.Id, Field.Store.YES),
                new TextField("subject", mail.Subject, Field.Store.YES),
                new TextField("from", mail.From, Field.Store.YES),
                new TextField("to", mail.To, Field.Store.YES),
                new StringField("date", mail.Date.ToString("yyyyMMdd"), Field.Store.YES),
                new TextField("body", mail.Body, Field.Store.YES),
                new TextField("all", $"{mail.Subject}\n{mail.From}\n{mail.To}\n{mail.Body}", Field.Store.NO)
            };

            writer.AddDocument(doc);
        }

        writer.Commit();
    }

    public IReadOnlyList<LuceneHit> Search(string queryText, int top = 5)
    {
        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var parser = new QueryParser(Version, "all", _analyzer)
        {
            DefaultOperator = Operator.OR
        };

        Query query;
        try
        {
            query = parser.Parse(QueryParserBase.Escape(queryText));
        }
        catch (ParseException)
        {
            query = new TermQuery(new Term("all", queryText.ToLowerInvariant()));
        }

        return searcher.Search(query, top).ScoreDocs
            .Select(hit =>
            {
                var doc = searcher.Doc(hit.Doc);
                return new LuceneHit(doc.Get("id"), hit.Score);
            })
            .ToArray();
    }

    public void Dispose()
    {
        _analyzer.Dispose();
        _directory.Dispose();
    }
}

public sealed record LuceneHit(string Id, float Score);
