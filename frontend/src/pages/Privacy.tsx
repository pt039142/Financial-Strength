import SectionedPage from "@/components/SectionedPage";

const PrivacyPage = () => {
  return (
    <SectionedPage
      eyebrow="Privacy"
      title="Privacy practices designed for enterprise trust"
      lead="We treat financial data carefully, with clear boundaries around access, storage, and retention."
      summary="This page is a product-facing overview of the privacy posture we expect to maintain as the platform matures."
      primaryCta={{ label: "Contact us", to: "/contact" }}
      secondaryCta={{ label: "Security", to: "/security" }}
      metrics={[
        { value: "Least", label: "Privilege by default" },
        { value: "Clear", label: "Access boundaries" },
        { value: "Scoped", label: "Data usage" },
        { value: "Audited", label: "Action history" },
      ]}
      highlights={[
        {
          title: "Data minimization",
          description: "Collect only the information needed to support the workflow being automated.",
        },
        {
          title: "Controlled retention",
          description: "Retain records only as long as required for operations, compliance, and customer needs.",
        },
        {
          title: "Transparency",
          description: "Keep customers informed about the kinds of data used and where it moves.",
        },
      ]}
      sections={[
        {
          eyebrow: "Policy",
          title: "Built to respect customer ownership of data",
          description: "Customer information should remain customer information, with platform access only where needed for the service.",
          bullets: [
            "Limit access to authorized users and service components.",
            "Keep sensitive operations traceable and reviewable.",
            "Use clear retention rules for operational records.",
          ],
        },
      ]}
    />
  );
};

export default PrivacyPage;
