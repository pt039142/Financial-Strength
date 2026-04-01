import SectionedPage from "@/components/SectionedPage";

const SecurityPage = () => {
  return (
    <SectionedPage
      eyebrow="Security"
      title="Security-first foundations for finance automation"
      lead="The platform touches sensitive operational data, so the architecture should assume trust is earned at every step."
      summary="This page outlines the intended direction for authentication, access control, storage, and auditing."
      primaryCta={{ label: "Talk to security", to: "/contact" }}
      secondaryCta={{ label: "Read privacy", to: "/privacy" }}
      metrics={[
        { value: "JWT", label: "Auth approach" },
        { value: "Role-based", label: "Access design" },
        { value: "Audit", label: "Activity trail" },
        { value: "Encrypted", label: "Sensitive fields" },
      ]}
      highlights={[
        {
          title: "Identity and access",
          description: "Use role-aware access control and verified tokens for all protected operations.",
        },
        {
          title: "Secure storage",
          description: "Keep financial files in object storage and avoid exposing sensitive data unnecessarily.",
        },
        {
          title: "Operational visibility",
          description: "Track important actions so teams can investigate and understand what happened.",
        },
      ]}
      sections={[
        {
          eyebrow: "Controls",
          title: "Security should be visible in the product, not hidden in a checklist",
          description: "A finance platform earns trust by making sensitive actions traceable and by reducing the blast radius of mistakes.",
          bullets: [
            "Authenticate every protected request.",
            "Scope access by organization and role.",
            "Log important events for audit review.",
          ],
        },
      ]}
    />
  );
};

export default SecurityPage;
