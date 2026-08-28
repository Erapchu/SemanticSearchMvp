using EFilerSemanticSearchMvp;

Console.OutputEncoding = System.Text.Encoding.UTF8;

var modelDir = ResolveModelDirectory();
var modelPath = Path.Combine(modelDir, "model_O4.onnx");
var tokenizerPath = Path.Combine(modelDir, "tokenizer.json");

if (!File.Exists(modelPath) || !File.Exists(tokenizerPath))
{
    Console.WriteLine("E5 model files are missing.");
    Console.WriteLine("Run download-model.ps1 from the project directory, then start again.");
    Console.WriteLine($"Expected: {modelPath}");
    return;
}

var messages = new[]
{
    new MailMessage(
        "1",
        "Birthday cake enquiry",
        "alice@bakery.example",
        "sean@efiler.example",
        new DateTime(2026, 8, 10),
        "Hi Sean, yes, we make bespoke birthday cakes to order. " +
        "Customers can choose the filling, decoration and delivery date. " +
        "Please send us the number of guests and the preferred design."),

    new MailMessage(
        "2",
        "SharePoint upload is very slow",
        "john@acme.example",
        "support@efiler.example",
        new DateTime(2026, 8, 11),
        "Since Monday our users have reported that large PDF files take several minutes " +
        "to upload to SharePoint. Small documents work normally. The issue appears only " +
        "when eFiler sync is running through the London office connection."),

    new MailMessage(
        "3",
        "Invoice for August",
        "accounts@vendor.example",
        "sean@efiler.example",
        new DateTime(2026, 8, 12),
        "Please find attached our August invoice for software maintenance. " +
        "Payment is due within thirty days. Bank details remain unchanged."),

    new MailMessage(
        "4",
        "Office celebration catering",
        "kate@events.example",
        "team@efiler.example",
        new DateTime(2026, 8, 13),
        "For the company celebration we can provide sandwiches, fruit and cupcakes. " +
        "The standard package is for twenty people. Custom desserts are available on request."),
};

using var lucene = new LuceneMailIndex();
lucene.Build(messages);

using var e5 = new E5EmbeddingService(modelPath, tokenizerPath);
var semantic = new SemanticMailIndex(e5);

Console.WriteLine("Creating 4 local E5 embeddings...");
semantic.Build(messages);
Console.WriteLine("Ready.\n");

var examples = new[]
{
    "тортики на заказ",
    "письмо где жаловались что файлы долго загружаются в SharePoint",
    "счёт за обслуживание",
    "еда для праздника в офисе"
};

foreach (var query in examples)
{
    Console.WriteLine(new string('=', 90));
    Console.WriteLine($"QUERY: {query}\n");

    var luceneHits = lucene.Search(query, top: 4);
    var semanticHits = semantic.Search(query, top: 4);
    var hybridHits = HybridSearch.Merge(luceneHits, semanticHits, top: 4);

    Print("Lucene", luceneHits.Select(x => (x.Id, (double)x.Score)));
    Print("E5 semantic", semanticHits.Select(x => (x.Id, (double)x.Score)));
    Print("Hybrid (RRF)", hybridHits.Select(x => (x.Id, x.Score)));

    var best = semanticHits.First();
    var message = messages.Single(x => x.Id == best.Id);
    Console.WriteLine($"Semantic winner: #{message.Id} — {message.Subject}");
    Console.WriteLine($"  {message.Body}\n");
}

static void Print(string title, IEnumerable<(string Id, double Score)> hits)
{
    Console.WriteLine($"{title}:");
    var any = false;
    foreach (var hit in hits)
    {
        any = true;
        Console.WriteLine($"  #{hit.Id,-2} score={hit.Score:F4}");
    }

    if (!any)
        Console.WriteLine("  <nothing found>");

    Console.WriteLine();
}

static string ResolveModelDirectory()
{
    var relative = Path.Combine("models", "multilingual-e5-small");
    var candidates = new[]
    {
        Path.Combine(Directory.GetCurrentDirectory(), relative),
        Path.Combine(AppContext.BaseDirectory, relative)
    };

    return candidates.FirstOrDefault(Directory.Exists) ?? candidates[0];
}
