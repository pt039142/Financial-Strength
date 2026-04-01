import SectionedPage from "@/components/SectionedPage";

const TermsPage = () => {
  return (
    <SectionedPage
      eyebrow="Terms"
      title="Straightforward terms for a product that handles financial workflows"
      lead="We want the platform to be easy to adopt, easy to understand, and easy to govern."
      summary="This route gives the website a real legal destination instead of a dead link, while the formal legal copy can be added later."
      primaryCta={{ label: "Contact us", to: "/contact" }}
      secondaryCta={{ label: "Compliance", to: "/compliance" }}
      metrics={[
        { value: "Clear", label: "Service expectations" },
        { value: "Fair", label: "Use boundaries" },
        { value: "Practical", label: "Operational rules" },
        { value: "Human", label: "Support focus" },
      ]}
      highlights={[
        {
          title: "Transparent usage",
          description: "The platform should make it obvious what is automated, what is reviewed, and what remains under customer control.",
        },
        {
          title: "Service continuity",
          description: "Operational commitments matter when the software sits inside finance processes.",
        },
        {
          title: "Responsible use",
          description: "Automation should improve workflows without creating hidden dependencies.",
        },
      ]}
      sections={[
        {
          eyebrow: "Agreement",
          title: "The product should be usable without legal guesswork",
          description: "The legal pages are intentionally lightweight here, giving the website a complete surface while the formal policy language evolves.",
          bullets: [
            "Describe the service clearly.",
            "Document limitations and support boundaries.",
            "Keep customer relationships easy to understand.",
          ],
        },
      ]}
    />
  );
};

export default TermsPage;
