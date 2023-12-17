export const formatDate = (date: string) => {
  const formatData = date.split("T");
  const year = formatData[0].split("-").reverse().join(".");
  const time = formatData[1].split(":").splice(0, 2).join(":");
  return `${year} ${time}`;
};
