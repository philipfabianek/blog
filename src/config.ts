export const SITE = {
  website: "https://philipfabianek.com/",
  author: "Philip Fabianek",
  profile: "https://github.com/philipfabianek",
  desc: "My personal blog.",
  title: "Philip Fabianek",
  ogImage: "og.png",
  lightAndDarkMode: true,
  postPerIndex: 10,
  postPerPage: 10,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: false,
  showBackButton: false,
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "https://github.com/philipfabianek/blog/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr",
  lang: "en",
  timezone: "Europe/Prague",
} as const;
