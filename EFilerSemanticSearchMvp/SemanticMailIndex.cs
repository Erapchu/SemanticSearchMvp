namespace EFilerSemanticSearchMvp;

public sealed class SemanticMailIndex
{
    private readonly E5EmbeddingService _embeddings;
    private readonly Dictionary<string, float[]> _vectors = new();

    public SemanticMailIndex(E5EmbeddingService embeddings)
    {
        _embeddings = embeddings;
    }

    public void Build(IEnumerable<MailMessage> messages)
    {
        foreach (var mail in messages)
            _vectors[mail.Id] = _embeddings.EmbedPassage(mail.SemanticText);
    }

    public IReadOnlyList<SemanticHit> Search(string query, int top = 5)
    {
        var queryVector = _embeddings.EmbedQuery(query);

        return _vectors
            .Select(pair => new SemanticHit(
                pair.Key,
                E5EmbeddingService.CosineSimilarity(queryVector, pair.Value)))
            .OrderByDescending(x => x.Score)
            .Take(top)
            .ToArray();
    }
}

public sealed record SemanticHit(string Id, float Score);
