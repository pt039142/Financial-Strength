import SectionedPage from "@/components/SectionedPage";

const SolutionsPage = () => {
  return (
    <SectionedPage
      eyebrow="Use Cases"
      title="Vertical solutions for document-heavy financial workflows"
      lead="Autonomiq can be shaped for small business funding, mortgage, tax, tenant screening, legal workflows, and more."
      summary="The platform is designed to be flexible enough for many verticals while keeping the core document-understanding and verification engine consistent."
      primaryCta={{ label: "Talk to Sales", to: "/contact" }}
      secondaryCta={{ label: "See Pricing", to: "/pricing" }}
      metrics={[
        { value: "SMB", label: "Funding workflows" },
        { value: "Mortgage", label: "Income verification" },
        { value: "Tax", label: "Document intake" },
        { value: "Legal", label: "Evidence review" },
      ]}
      highlights={[
        {
          title: "Small business funding",
          description: "Speed up underwriting with cash flow analysis and structured document capture.",
        },
        {
          title: "Mortgage",
          description: "Evaluate income and conditions faster with document understanding and review workflows.",
        },
        {
          title: "Tenant screening and tax",
          description: "Standardize document-heavy verification flows across tenants, applicants, and tax-related use cases.",
        },
      ]}
      sections={[
        {
          eyebrow: "Vertical",
          title: "Industry-specific workflows with the same core intelligence",
          description: "Different businesses need different outputs, but the underlying engine for capture, verification, and review stays consistent.",
          bullets: [
            "Reuse the same document understanding pipeline across multiple workflows.",
            "Tune review logic by vertical, not just by document type.",
            "Keep decisioning structured and auditable.",
          ],
        },
        {
          eyebrow: "Vertical",
          title: "Developer-friendly integrations for downstream systems",
          description: "Structured outputs can be pushed into lending systems, CRMs, accounting workflows, or internal operations dashboards.",
          bullets: [
            "Webhooks and APIs for processed document events.",
            "Normalized JSON payloads for downstream consumption.",
            "Approval and exception routing for high-risk submissions.",
          ],
        },
      ]}
    />
  );
};

export default SolutionsPage;
