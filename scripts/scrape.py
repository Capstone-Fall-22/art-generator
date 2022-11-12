import json, os, re, requests, shutil, sys
from bs4 import BeautifulSoup

WIKIMEDIA_URL = "https://commons.wikimedia.org"


def get_soup(link: str, parser: str = "lxml"):
    html = requests.get(link).text
    return BeautifulSoup(html, parser)


def get_navigation_links(
    initial_link: str, link_text: str, link_prefix: str = ""
) -> list:
    """
    link_string should be unique to a single <a> tag on the page
    """
    links = [initial_link]

    try:
        soup = get_soup(initial_link)
        while soup:
            a_tag = soup.find("a", string=link_text)
            link = link_prefix + a_tag["href"]
            links.append(link)
            soup = get_soup(link)
    except:
        pass

    return links


def get_image_links(
    navigation_links: list, image_link_class: str, link_prefix: str = ""
):
    image_links = list()
    for link in navigation_links:
        soup = get_soup(link)
        links = soup.find_all("a", class_=image_link_class)
        links = [link_prefix + link["href"] for link in links]
        image_links.extend(links)
    return image_links


def shorten_file_name(file_name: str):
    file_name = file_name.split(".")
    file_extension = file_name.pop()
    file_name = ".".join(file_name)

    # Avoid file name too long error
    # Shorten the file name to be 255 (after adding file extension with period: .jpg or .png)
    file_name = file_name[: 255 - len(file_extension) - 1]
    file_name = f"{file_name}.{file_extension}"
    return file_name


def download_images(image_links: list, output_dir: str):
    acceptible_licenses = [
        re.compile(f"Attribution-Share Alike {i}") for i in [2.0, 2.5, 3.0, 4.0]
    ]

    attributions = dict()

    with open(os.path.join(output_dir, "attributions.json"), "w") as attributions_file:
        for i, link in enumerate(image_links):
            # if i == 4: # For quick testing
            #     break

            print(f"Downloading image #{i + 1}/{len(image_links)}", end="\r")
            soup = get_soup(link)

            # Skip images with licenses that are not compatible with CC BY-SA 4.0
            valid_license = False
            for license in acceptible_licenses:
                if soup.find("a", string=license):
                    valid_license = True
                    break

            if not valid_license:
                continue

            try:
                download_link = soup.find("div", class_="fullImageLink").find("a")[
                    "href"
                ]
            except:
                continue
            author_name = soup.find("a", class_="mw-userlink").text
            output_file_name = download_link.split("/")[-1]
            if len(output_file_name) > 255:
                output_file_name = shorten_file_name(output_file_name)

            headers = {"User-Agent": "School Project Bot 1.0"}
            image = requests.get(download_link, headers=headers).content

            with open(os.path.join(output_dir, output_file_name), "wb") as image_file:
                image_file.write(image)

            attributions[output_file_name] = author_name
            attributions_file.seek(0)
            json.dump(attributions, attributions_file, indent=4)


def category_exists(category):
    res = requests.get(WIKIMEDIA_URL + "/wiki/Category:" + category)
    if res.status_code == 200:
        return True

    return False


def get_all_category_urls(category: str, max_depth: int) -> list:
    if max_depth == 0:
        return [WIKIMEDIA_URL + "/wiki/Category:" + category]

    queue = [WIKIMEDIA_URL + "/wiki/Category:" + category]

    all_category_urls = list()

    for i in range(max_depth):
        new_queue = list()
        while len(queue):
            category_url = queue.pop()
            all_category_urls.append(category_url)

            soup = get_soup(category_url)

            # Get subcategories section
            for subcategory in soup.find_all("div", class_="CategoryTreeItem"):
                new_queue.append(WIKIMEDIA_URL + subcategory.find("a")["href"])

        queue = new_queue

    all_category_urls.extend(queue)

    return all_category_urls


def aggregate_attributions(output_dir):
    attributions = dict()
    for category in os.listdir(output_dir):
        # cannot aggregate attributions.json file in top level directory, must be in category directory
        if category == "attributions.json":
            continue

        with open(
            os.path.join(output_dir, category, "attributions.json"), "r"
        ) as attributions_file:
            # If file in empty, no images were downloaded for that category
            # This can be due to the subcategories not having any images with
            # licenses that are compatible with CC BY-SA 4.0
            if (
                os.stat(os.path.join(output_dir, category, "attributions.json")).st_size
                == 0
            ):
                continue
            attributions[category] = json.load(attributions_file)

    with open(os.path.join(output_dir, "attributions.json"), "w") as attributions_file:
        json.dump(attributions, attributions_file, indent=4)


def cleanup_incomplete(output_dir):
    for category in os.listdir(output_dir):
        if category.endswith("_*incomplete"):
            shutil.rmtree(os.path.join(output_dir, category))


def scrape(category_urls, output_dir, max_categories=0):
    cleanup_incomplete(output_dir)

    if max_categories:
        category_urls = category_urls[:max_categories]

    for i, category_url in enumerate(category_urls):
        navigation_links = get_navigation_links(
            category_url, "next page", WIKIMEDIA_URL
        )
        image_links = get_image_links(
            navigation_links, "galleryfilename", WIKIMEDIA_URL
        )

        category_name = category_url.split(":")[-1]
        category_name = category_name.replace("/", "_")
        category_dir = os.path.join(output_dir, category_name)
        if os.path.exists(category_dir):  # skip category if already downloaded
            continue

        category_dir += "_*incomplete"

        os.mkdir(category_dir)

        os.system("cls" if os.name == "nt" else "clear")  # Clear terminal
        print(f"Scraping category #{i + 1}/{len(category_urls)}: {category_name}")

        download_images(image_links, category_dir)

        # remove _*incomplete from directory name to indicate completion
        os.rename(category_dir, category_dir[:-12])

    aggregate_attributions(output_dir)


def main(*args, **kwargs):
    category = input("Enter category to scrape\n> ")
    if not category_exists(category):
        sys.exit("Category doesn't exist")

    subcategory_depth = input(
        "How many levels of subcategories to scrape? (Recommended: 2)\n> "
    )
    if not subcategory_depth.isdigit():
        sys.exit("Please try again with an integer")
    subcategory_depth = int(subcategory_depth)

    max_categories = input(
        "What is the maximum number of categories to scrape (put 0 for no limit)\n> "
    )
    if not max_categories.isdigit():
        sys.exit("Please try again with an integer")
    max_categories = int(max_categories)

    output_dir = "./" + input("Enter name of directory to save images in?\n> ")
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            sys.exit("Please try again with another directory name")

    category_urls = get_all_category_urls(category, subcategory_depth)

    scrape(category_urls, output_dir, max_categories)
    return 0


if __name__ == "__main__":
    main()
