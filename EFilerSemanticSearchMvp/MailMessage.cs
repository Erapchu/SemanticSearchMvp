namespace EFilerSemanticSearchMvp;

public sealed record MailMessage(
    string Id,
    string Subject,
    string From,
    string To,
    DateTime Date,
    string Body)
{
    public string SemanticText =>
        $"Subject: {Subject}\nFrom: {From}\nTo: {To}\nDate: {Date:yyyy-MM-dd}\n\n{Body}";
}
