import SectionedPage from "@/components/SectionedPage";

const AboutPage = () => {
  return (
    <SectionedPage
      eyebrow="About"
      title="We are building the operating system for autonomous finance"
      lead="Autonomiq exists to replace fragmented, manual finance operations with a coherent AI-first system that scales with the business."
      summary="The goal is simple: give finance teams speed without sacrificing accuracy, controls, or accountability."
      primaryCta={{ label: "Meet the Team", to: "/contact" }}
      secondaryCta={{ label: "Explore Products", to: "/products" }}
      metrics={[
        { value: "Mission", label: "Finance automation first" },
        { value: "Design", label: "Human-in-the-loop" },
        { value: "Focus", label: "Auditability" },
        { value: "Goal", label: "Operational clarity" },
      ]}
      highlights={[
        {
          title: "Built for trust",
          description: "Every workflow is designed around control, traceability, and reviewability instead of black-box automation.",
        },
        {
          title: "Built for scale",
          description: "The architecture anticipates multi-tenant usage, asynchronous processing, and integrations across the finance stack.",
        },
        {
          title: "Built for teams",
          description: "We care about helping operators work faster without forcing them to change the tools they already rely on.",
        },
      ]}
      sections={[
        {
          eyebrow: "Mission",
          title: "Remove the bottlenecks that slow finance teams down",
          description: "Finance should be a strategic function, not a pile of repetitive data entry and spreadsheet reconciliation.",
          bullets: [
            "Automate the repetitive parts of finance workflows.",
            "Keep humans in control of the final decisions.",
            "Deliver systems that are explainable and auditable.",
          ],
        },
        {
          eyebrow: "Vision",
          title: "Make finance operations feel like infrastructure, not manual work",
          description: "The long-term vision is a platform where financial processes run continuously in the background and surface exceptions clearly.",
          bullets: [
            "Live reporting with fewer handoffs.",
            "Unified data for bookkeeping, reporting, and compliance.",
            "A platform that grows with startups and enterprises alike.",
          ],
        },
      ]}
    />
  );
};

export default AboutPage;
