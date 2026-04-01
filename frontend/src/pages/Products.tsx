import SectionedPage from "@/components/SectionedPage";

const ProductsPage = () => {
  return (
    <SectionedPage
      eyebrow="Platform"
      title="AI engine for document and decision intelligence"
      lead="Autonomiq analyzes financial documents with speed and precision, then routes uncertain cases to human reviewers when confidence is not enough."
      summary="This is the control layer for document understanding, fraud detection, and workflow automation, built for high-volume finance teams."
      primaryCta={{ label: "Request Demo", to: "/contact" }}
      secondaryCta={{ label: "Explore Use Cases", to: "/solutions" }}
      metrics={[
        { value: "99+%", label: "Document accuracy" },
        { value: "24/7", label: "Workflow coverage" },
        { value: "6+", label: "Vertical workflows" },
        { value: "1", label: "Unified platform" },
      ]}
      highlights={[
        {
          title: "Document understanding",
          description: "Specialized capture for bank statements, pay stubs, invoices, and tax documents.",
        },
        {
          title: "Fraud detection",
          description: "Flag tampering, inconsistencies, and suspicious patterns before bad data moves downstream.",
        },
        {
          title: "Human-in-the-loop review",
          description: "Route low-confidence items to reviewers so edge cases are resolved with auditability and control.",
        },
      ]}
      sections={[
        {
          eyebrow: "Capability",
          title: "Document processing that feels built for scale",
          description: "The platform organizes document-heavy work into a repeatable flow: ingest, classify, capture, verify, and deliver structured output.",
          bullets: [
            "Capture and classify incoming files from multiple sources.",
            "Extract structured fields from statements, forms, and invoices.",
            "Maintain audit-ready review trails for every exception.",
          ],
        },
        {
          eyebrow: "Capability",
          title: "Decisioning workflows for underwriting and operations",
          description: "Use the platform to calculate cash flow, assess income, and route cases through review before a decision is made.",
          bullets: [
            "Measure cash flow and income across financial documents.",
            "Use confidence scoring to prioritize human review.",
            "Push structured outputs into lending or operations systems.",
          ],
        },
      ]}
    />
  );
};

export default ProductsPage;
