import SectionedPage from "@/components/SectionedPage";

const CompliancePage = () => {
  return (
    <SectionedPage
      eyebrow="Compliance"
      title="Compliance workflows that keep finance teams audit-ready"
      lead="Automation is most valuable when it supports controls instead of bypassing them."
      summary="Autonomiq is structured around reviewable workflows, approval paths, and records that help teams demonstrate what happened."
      primaryCta={{ label: "Discuss rollout", to: "/contact" }}
      secondaryCta={{ label: "Explore solutions", to: "/solutions" }}
      metrics={[
        { value: "Reviewable", label: "Outputs" },
        { value: "Traceable", label: "Actions" },
        { value: "Structured", label: "Records" },
        { value: "Ready", label: "For audits" },
      ]}
      highlights={[
        {
          title: "Approval flows",
          description: "Keep human review in the loop where the business needs judgment instead of pure automation.",
        },
        {
          title: "Audit logs",
          description: "Capture the actions, changes, and decisions that matter for finance operations.",
        },
        {
          title: "Reporting support",
          description: "Structure outputs so the data can be reviewed, shared, and traced back to source records.",
        },
      ]}
      sections={[
        {
          eyebrow: "Approach",
          title: "Automation with guardrails",
          description: "Compliance is easier when the product makes it simpler to approve, review, and explain the work.",
          bullets: [
            "Human review for uncertain outcomes.",
            "Consistent records across workflows.",
            "A platform that supports control owners.",
          ],
        },
      ]}
    />
  );
};

export default CompliancePage;
