        digit_w, digit_h = 60, 110

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
                if h / result.shape[0] >= 0.3:  # Select contour which has the height larger than 30% of the plate
                    # Draw bounding box arroung digit number
                    w = w + 10
                    h = h + 10

                    cv2.rectangle(test_roi, (x - 5, y - 5), (x + w, y + h), (0, 255, 0), 2)
                    # print(x,y,w,h)

                    # Sperate number and gibe prediction
                    curr_num = gray[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    # _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)