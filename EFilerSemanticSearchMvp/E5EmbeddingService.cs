using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.HuggingFace.Tokenizer;

namespace EFilerSemanticSearchMvp;

public sealed class E5EmbeddingService : IDisposable
{
    public const int Dimensions = 384;
    public const int MaxTokens = 512;

    private readonly Tokenizer _tokenizer;
    private readonly InferenceSession _session;

    public E5EmbeddingService(string modelPath, string tokenizerPath)
    {
        _tokenizer = Tokenizer.FromFile(tokenizerPath);
        _session = new InferenceSession(modelPath);
    }

    public float[] EmbedQuery(string text) => Embed("query: " + text);

    public float[] EmbedPassage(string text) => Embed("passage: " + text);

    private float[] Embed(string text)
    {
        var encoding = _tokenizer.Encode(text, true, includeTypeIds: true, includeAttentionMask: true).First();

        // MVP: truncate to the model's maximum context length.
        // Production eFiler should chunk long emails before embedding.
        var length = Math.Min(encoding.Ids.Count, MaxTokens);

        var ids = encoding.Ids.Take(length).Select(x => (long)x).ToArray();
        var typeIds = encoding.TypeIds.Take(length).Select(x => (long)x).ToArray();
        var attentionMask = encoding.AttentionMask.Take(length).Select(x => (long)x).ToArray();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(ids, new[] { 1, length })),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(attentionMask, new[] { 1, length }))
        };

        // The official ONNX export currently exposes token_type_ids.
        // Keeping the check makes this resilient to a future export that omits it.
        if (_session.InputMetadata.ContainsKey("token_type_ids"))
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                "token_type_ids",
                new DenseTensor<long>(typeIds, new[] { 1, length })));
        }

        using var results = _session.Run(inputs);
        var hidden = results.First().AsTensor<float>();

        var embedding = new float[Dimensions];
        var weight = 0f;

        // E5 model card specifies average pooling over non-padding tokens.
        for (var token = 0; token < length; token++)
        {
            if (attentionMask[token] == 0)
                continue;

            weight += 1f;
            for (var dim = 0; dim < Dimensions; dim++)
                embedding[dim] += hidden[0, token, dim];
        }

        if (weight == 0)
            return embedding;

        for (var dim = 0; dim < Dimensions; dim++)
            embedding[dim] /= weight;

        NormalizeL2(embedding);
        return embedding;
    }

    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vector sizes must match.");

        // Embeddings are already L2-normalized, so cosine == dot product.
        var sum = 0f;
        for (var i = 0; i < a.Length; i++)
            sum += a[i] * b[i];

        return sum;
    }

    private static void NormalizeL2(Span<float> vector)
    {
        double sumSquares = 0;
        foreach (var value in vector)
            sumSquares += value * value;

        var norm = Math.Sqrt(sumSquares);
        if (norm == 0)
            return;

        for (var i = 0; i < vector.Length; i++)
            vector[i] = (float)(vector[i] / norm);
    }

    public void Dispose()
    {
        _session.Dispose();
        _tokenizer.Dispose();
    }
}
