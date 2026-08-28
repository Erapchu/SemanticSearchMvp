namespace EFilerSemanticSearchMvp;

public static class HybridSearch
{
    public static IReadOnlyList<HybridHit> Merge(
        IReadOnlyList<LuceneHit> lucene,
        IReadOnlyList<SemanticHit> semantic,
        int top = 5)
    {
        // Reciprocal Rank Fusion: combines rankings without trying to compare
        // Lucene BM25 scores with cosine scores directly.
        const double k = 60;
        var scores = new Dictionary<string, double>();

        for (var i = 0; i < lucene.Count; i++)
            scores[lucene[i].Id] = scores.GetValueOrDefault(lucene[i].Id) + 1.0 / (k + i + 1);

        for (var i = 0; i < semantic.Count; i++)
            scores[semantic[i].Id] = scores.GetValueOrDefault(semantic[i].Id) + 1.0 / (k + i + 1);

        return scores
            .OrderByDescending(x => x.Value)
            .Take(top)
            .Select(x => new HybridHit(x.Key, x.Value))
            .ToArray();
    }
}

public sealed record HybridHit(string Id, double Score);
